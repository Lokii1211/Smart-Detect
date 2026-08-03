# Multi-Camera Scale Architecture — Design

**Status:** DESIGN ONLY — nothing here is implemented.
**Date:** 2026-08-02
**Author:** engineering
**Grounding:** every number below was measured on this machine (Apple M5, 16 GB, macOS 26) with `scripts/bench_scale.py` and `eval/profile_pipeline.py`. Where a number is estimated rather than measured, it says so.

---

## 0. Executive summary

Three findings changed this design relative to what was assumed in `PROJECT_REVIEW.md` §5.3:

1. 🔴 **The pgvector path does not work and never has.** It has two independent defects and would crash on the first identity lookup against PostgreSQL. Item 3 was supposed to "execute the untested path" — it did, and the path is broken.
2. 🟢 **SQLite is not the write bottleneck.** At 16 concurrent writers it sustained 10,203 inserts/s with zero errors. The real sighting rate is ~1 write per code per 30 s. The case for PostgreSQL is **tail latency and multi-host distribution**, not throughput. My earlier "SQLite will thrash" claim was wrong.
3. 🔴 **Memory, not CPU, caps process-per-camera.** A fully-loaded pipeline process is **1,459 MB** resident, of which InsightFace alone is 1,045 MB. On this 16 GB box that is ~9 processes — and the CoreML work already got one process to 14.5 fps analysis. Process-per-camera trades 1.4 GB per camera for GIL isolation.

**Recommended order:** fix pgvector (P0, ~4 d) → HNSW index (P0, ~2 d) → worker pool (P1, ~10 d) → broker (P2, ~7 d). Total ~23 engineer-days to a design that holds 15–20 cameras.

---

## 1. Where we are now

After the CoreML/lazy-re-ID work (`docs/` perf results):

| Metric | Value | Source |
|---|---|---|
| Per-frame analysis | 69.1 ms → **14.46 fps** | `eval/profile_pipeline.py` |
| Face stage (InsightFace) | 44.4 ms (60%) | measured |
| Detect (YOLO) | 10.6 ms (14%) | measured |
| Re-ID (OSNet, amortised) | 6.5 ms (9%) | measured |
| Concurrent camera cap | 4 (`MAX_STREAMS`) | code constant |
| Process RSS, models loaded | **1,459 MB** | measured |
| Identity search @ 15 identities | <1 ms | trivial today |

The architecture is threads inside one process: one capture thread + one analysis thread per camera, sharing a `Queue(maxsize=1)` (already an adaptive frame-skipper), all under one GIL.

---

## 2. Item 1 — Process-per-camera or worker pool

### The actual constraint

Not the GIL alone. The pipeline spends its time inside **native code that releases the GIL** — onnxruntime, PyTorch, OpenCV. So threads already parallelise the expensive parts to a degree. The measured limits are:

- **Memory: 1,459 MB/process.** ~9 processes fit in 16 GB. This is the binding constraint.
- **Python-level contention:** arbitration, tracking, box maths, and DB session work do hold the GIL.

### Recommendation: **worker pool, not process-per-camera**

Process-per-camera means N × 1.4 GB, mostly duplicated model weights. A pool of `W` analysis workers fed by a shared queue decouples worker count from camera count:

```
capture procs (thin, ~80 MB each)        analysis pool (heavy, 1.4 GB each)
  cam-01 ──┐                              ┌── worker-1 ──┐
  cam-02 ──┼──> frame queue (latest-wins) ┼── worker-2 ──┼──> results bus ──> API/DB
  cam-NN ──┘        per camera            └── worker-W ──┘
```

- Capture processes hold no models: decode + JPEG-encode for MJPEG only.
- Workers are stateless w.r.t. *frames* but **not** w.r.t. *tracks*. This is the hard part (§2.1).
- `W` sized by memory: `W = floor((RAM − overhead) / 1.5 GB)`. On 16 GB, `W = 8`.
- Capacity ≈ `W × 14.5` analysis-fps, distributed across cameras by demand.

### 2.1 The hard problem: per-track state is not shareable

`LiveStream` holds `_track_codes`, `_track_age`, `_track_face_mismatch`, `_seen_cache`, and a `ByteTrack` instance — all mutable per-camera state. The ID-switch guard is **stateful across frames** (`id_switch_contradiction_limit` counts *consecutive* contradictions).

If two workers process consecutive frames of one camera, that state is split and the guard silently stops working — a correctness regression that the ablation would catch only if the dataset were adequate.

**Mitigation: sticky routing.** Hash `camera_id` → worker. One camera is always handled by the same worker, so per-track state stays process-local and ByteTrack stays coherent. Cost: load imbalance if cameras differ in busyness; rebalancing requires draining a camera's state.

**Rejected alternative:** externalising track state to Redis. Adds a network round-trip inside the per-frame hot path (~69 ms budget) and makes the ID-switch guard's read-modify-write racy. Not worth it.

### Effort

| Task | Days |
|---|---|
| Extract analysis core from `LiveStream` into a frame-in/results-out callable | 3 |
| Capture process + shared-memory frame transport (avoid pickling 1 MB frames) | 2 |
| Worker pool + sticky `camera_id` routing + supervision/restart | 3 |
| MJPEG serving from capture process (annotate with last results) | 2 |
| Health/metrics, graceful drain, backpressure | 2 |
| Test: verify ablation unchanged, soak at W=8 | 2 |
| **Total** | **14 d** |

Risk: **high**. Touches the most stateful code in the system. Requires the adequate eval dataset first, or regressions will be invisible.

---

## 3. Item 2 — Redis / RabbitMQ between capture and analysis

### Recommendation: **defer. Not justified yet.**

Today capture→analysis is an in-process `Queue(maxsize=1)` with latest-wins semantics — the correct policy for live video (a stale frame is worthless). A broker adds value only when capture and analysis are on **different hosts**.

Measured argument against premature adoption: a 1 MB frame at 30 fps per camera is ~30 MB/s/camera. Through Redis that is a serialize → network → deserialize round trip inside a 69 ms budget. For 8 cameras that is 240 MB/s of broker traffic to move data between processes that could share memory.

**When it becomes right:** edge capture nodes + a central GPU inference box. Then send *JPEG-compressed* frames (~60 KB, 20× smaller), accepting the encode/decode cost.

### If/when built

| Choice | Verdict |
|---|---|
| **Redis Streams** | ✅ Recommended. Consumer groups give sticky routing and at-least-once; `MAXLEN ~ N` caps memory and drops old frames naturally — matching latest-wins. |
| RabbitMQ | ❌ Per-message ack overhead and durability guarantees are wasted on frames we want to *drop*. |
| Kafka | ❌ Retention/replay semantics are the opposite of what live video needs. |

**Critical design rule:** frames must be **droppable**. A queue that buffers under load converts a throughput problem into an unbounded-latency problem — boxes drift further behind reality the more overloaded you are. Use `XADD ... MAXLEN ~ 2`.

### Effort

| Task | Days |
|---|---|
| Redis Streams transport + consumer groups | 3 |
| JPEG frame codec + drop policy + backpressure | 2 |
| Deploy/ops: Redis HA, monitoring, failure modes | 2 |
| **Total** | **7 d** (only after item 1) |

---

## 4. Item 3 — PostgreSQL + pgvector — **EXECUTED, and it is broken**

I stood up PostgreSQL 17.10 + pgvector 0.8.6 and ran the shipped `_find_person_pgvector` for the first time.

### 🔴 Defect 1 — the query cannot bind its parameter (fatal)

```python
# database/queries.py:82
1 - ({col}::vector <=> :vec::vector) AS similarity
```

SQLAlchemy's `text()` bind-parser mis-reads `::`. Measured:

```
params SQLAlchemy sees in the PRODUCTION statement: ['ve']
params with CAST(... AS vector)                   : ['vec']
```

It extracts a parameter named **`ve`**, not `vec`. The code binds `{"vec": ...}`, so `:vec` is never substituted and raw SQL reaches the server:

```
psycopg2.errors.SyntaxError: syntax error at or near ":"
LINE 3:  1 - (face_embedding::vector <=> :vec::vector)...
```

**Any deployment switching `DATABASE_URL` to PostgreSQL crashes on the first identity lookup.** The `_is_postgres(db)` branch is entered correctly; the query inside it has never run.

**Fix:** `CAST(:vec AS vector)`. Verified working — returned the correct identity `SDT-000123` at similarity 0.9972, exactly matching the Python path's `0.99716`.

### 🔴 Defect 2 — the column cast defeats the index (silent, worse)

Even with defect 1 fixed, `CAST(face_embedding AS vector)` on the **column** makes `ORDER BY` a non-indexable expression:

```
->  Seq Scan on persons (actual time=0.071..36.782 rows=10000)
```

An HNSW index exists and is ignored. Fixing defect 1 alone yields a working-but-unscalable system: 289 ms/query at 100k. The column must be `vector(512)` in the schema and **must not be cast** in the query.

The root cause is schema drift: `models.py` declares `face_embedding = Column(Text)` because SQLite has no vector type. PostgreSQL needs a real `vector(512)` column.

### Measured: concurrent writers

100 inserts per writer, each its own transaction:

| Backend | Writers | Throughput | p50 | p99 | Errors |
|---|---|---|---|---|---|
| SQLite/WAL | 1 | 5,824/s | 0.13 ms | 0.69 ms | 0 |
| SQLite/WAL | 4 | 4,845/s | 0.10 ms | 1.02 ms | 0 |
| SQLite/WAL | 8 | 4,810/s | 0.08 ms | 4.09 ms | 0 |
| SQLite/WAL | 16 | 10,203/s | 0.06 ms | **23.93 ms** | 0 |
| PostgreSQL | 1 | 4,182/s | 0.22 ms | 0.37 ms | 0 |
| PostgreSQL | 4 | 16,564/s | 0.22 ms | 0.36 ms | 0 |
| PostgreSQL | 8 | 17,160/s | 0.39 ms | 1.10 ms | 0 |
| PostgreSQL | 16 | 16,758/s | 0.76 ms | **2.69 ms** | 0 |

**SQLite did not fall over.** Zero errors at 16 writers. But p99 degrades **35×** (0.69 → 23.93 ms) as writers contend for the single write lock, while PostgreSQL holds p99 under 3 ms and scales throughput 4×.

**Honest conclusion:** at SmartDetect's real write rate (≈1 sighting per code per 30 s — under 1 write/s even at 16 cameras) **SQLite is adequate**. Migrate for:
- **Vector search** (the actual reason — see item 4),
- **Multi-host access** — SQLite cannot be shared across machines, which blocks distributed capture,
- **Tail latency** under mixed read/write load.

Not for write throughput. That was my earlier error.

### Effort

| Task | Days |
|---|---|
| Fix defect 1 (`CAST(:vec AS vector)`) + regression test | 0.5 |
| Schema: dialect-aware `vector(512)` vs `Text`, Alembic migration | 2 |
| Fix defect 2 (drop column cast) + `EXPLAIN` assertion in tests | 0.5 |
| Data migration SQLite → PostgreSQL (`scripts/supabase_migrate.py` exists, untested) | 1.5 |
| Ops: connection pooling, backups, `pg_isready` health | 1.5 |
| **Total** | **6 d** |

Risk: **low-medium**. Bugs are understood and the fixes are verified.

---

## 5. Item 4 — HNSW replacing the linear scan

`_find_person_python` pulls every row, `json.loads` each vector, and cosines in a Python loop.

### Measured — 512-d, realistic probes (query = stored vector perturbed to 0.85 cosine)

| Identities | Python scan (current) | numpy matmul | hnswlib | pgvector exact | pgvector + HNSW |
|---|---|---|---|---|---|
| 1,000 | 0.97 ms | 0.01 ms | 0.35 ms | 3.40 ms | 2.51 ms † |
| 10,000 | 8.78 ms | 0.22 ms | 0.31 ms | 22.12 ms | **2.08 ms** |
| 100,000 | **91.10 ms** | 2.79 ms | **0.41 ms** | 289.08 ms | **3.59 ms** |

† planner correctly chose a seq scan — the table is too small to index. Not a bug.

Recall@1 (hnswlib, ef=64): 1.000 @ 1k, 1.000 @ 10k, **0.850 @ 100k**.
Build: hnswlib 37.6 s @ 100k; pgvector HNSW 108.9 s @ 100k.

> **Methodology note.** An earlier run used random-vs-random probes and reported recall 0.35. That is the pathological case: random 512-d vectors are near-orthogonal, so "nearest" is decided by noise. Real face matching probes a *perturbed copy* of a stored vector. With realistic probes recall is 1.000 at ≤10k. **Do not benchmark ANN with random queries.**

### Analysis

- The linear scan is **fine to ~1k** (0.97 ms against a 69 ms frame budget).
- At 10k it is 8.78 ms — 13% of the budget. Marginal.
- At 100k it is **91 ms — larger than the entire rest of the pipeline.** Unusable.
- **`numpy` matmul is the highest value-per-effort fix by far**: 0.22 ms @ 10k, 2.79 ms @ 100k, **exact** (no recall loss), ~half a day, no new dependency, no index to maintain. It replaces a Python loop with one BLAS call.
- ANN only wins above ~100k, where it must be weighed against recall loss (0.85 @ 100k).

### Recommendation — staged

1. **Now (≤10k identities): vectorise the scan with numpy.** 0.5 d, exact, ~40× faster. Do this regardless of the database decision.
2. **At >10k: pgvector HNSW.** 2.08–3.59 ms, index maintained by the DB, no separate sync. Requires item 3 fixed first.
3. **Only if >1M or sub-ms required: hnswlib in-process.** 0.41 ms, but you own index build, persistence, and **invalidation on every erasure** — which now happens routinely via the governance purge. An in-process index that drifts from the DB after a `DELETE /persons/{code}` is a compliance defect, not just a bug.

### Effort

| Task | Days |
|---|---|
| numpy-vectorised exact scan + equivalence test vs current | 1 |
| pgvector HNSW index + `EXPLAIN`-asserting test + `ef_search` tuning | 1.5 |
| (optional) hnswlib service, persistence, purge-driven invalidation | 5 |
| **Total (stages 1–2)** | **2.5 d** |

---

## 6. Item 5 — Migration path

Sequenced so each step is independently shippable and reversible. **No step depends on a later one.**

### Phase 0 — prerequisite (blocking)
**Acquire the labelled dataset.** The eval harness fails its own adequacy gate (2 identities, 45 frames). Every step below risks silent accuracy regression that current data cannot detect. ~1 d labelling with `eval/label_tool.py`. **Do not start Phase 2+ without this.**

### Phase 1 — cheap wins, no architecture change (3 d)
1. numpy-vectorised scan (0.5 d) — 40× on identity search, exact.
2. Fix pgvector defects 1 & 2 (1 d) — makes the path *work*; nothing switches to it yet.
3. Raise `MAX_STREAMS` 4 → 8 and soak-test (0.5 d) — CoreML made this plausible; verify against memory and p99, do not assume.
4. Re-run ablation after each (1 d).

*Rollback: revert commit. No data or schema change.*

### Phase 2 — PostgreSQL (6 d)
5. Dialect-aware schema (`vector(512)` on PG, `Text` on SQLite).
6. Migrate data; run **both** backends in parallel and diff identity decisions on the same input.
7. Cut over behind `DATABASE_URL`. Keep SQLite working — it is the right choice for a single-camera pilot.
8. Add HNSW index once >10k identities.

*Rollback: point `DATABASE_URL` back at SQLite. Keep the SQLite file until confident.*

### Phase 3 — worker pool (14 d)
9. Extract the analysis core; **prove equivalence on the ablation before changing the process model.**
10. Introduce the pool with `W=1` — identical behaviour, new plumbing.
11. Scale `W` up, verify ablation unchanged at each step.
12. Sticky routing; explicitly test ID-switch-guard continuity across many frames.

*Rollback: `W=1` and the in-process path is behaviourally the old system. Keep a feature flag.*

### Phase 4 — distribution (7 d, only if multi-host)
13. Redis Streams with drop-on-overflow.
14. Split capture from analysis across hosts.

### Non-goals
- **Do not** batch face inference across cameras: `det_10g.onnx` has batch fixed at 1, and detection is 73% of the face stage. Measured; not worth the coordinator.
- **Do not** externalise per-track state to Redis (§2.1).
- **Do not** adopt Kafka.

---

## 7. Effort summary

| Item | Days | Risk | Priority |
|---|---|---|---|
| Phase 0 — labelled dataset | 1 | low | **P0 — blocks everything** |
| 4a. numpy vectorised scan | 1 | low | **P0** — 40×, exact, trivial |
| 3. Fix pgvector defects | 1 | low | **P0** — path is broken today |
| 3. PostgreSQL migration | 5 | med | P1 |
| 4b. pgvector HNSW index | 1.5 | low | P1 (at >10k identities) |
| 1. Worker pool | 14 | **high** | P1 |
| 2. Redis Streams | 7 | med | P2 (multi-host only) |
| 4c. hnswlib service | 5 | high | P3 (>1M only) |
| **Total P0–P1** | **~23.5 d** | | |

### Projected capacity (estimate, not measured)

| Config | Cameras | Basis |
|---|---|---|
| Today | 4 | `MAX_STREAMS`, 14.5 fps analysis |
| Phase 1 | ~8 | memory-bound: 1.46 GB × 8 ≈ 11.7 GB |
| Phase 3, 16 GB | ~8–9 | **memory-capped**, not CPU |
| Phase 3, 64 GB | ~25–30 | W≈40 by RAM; CPU becomes the limit |
| Phase 4 + GPU | 40+ | unvalidated — a hypothesis to test |

> These are extrapolations from single-process measurements. **Treat as hypotheses.** The only honest way to know is to build Phase 3 with `W=1`, then measure at each `W`.

---

## 8. Reproducing the measurements

```bash
brew install postgresql@17 pgvector && brew services start postgresql@17
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
createdb smartdetect_scale && psql smartdetect_scale -c "CREATE EXTENSION vector;"
pip install hnswlib

python scripts/bench_scale.py --pg "postgresql://$USER@localhost/smartdetect_scale" --all \
    --json-out eval/results/perf/scale.json
python eval/profile_pipeline.py --video demo_videos/05_head_pose_two_people.mp4 --frames 60
```

Raw results: `eval/results/perf/scale_search.json`, `scale_writers.json`, `baseline.json`, `opt1_coreml.json`, `opt3_lazy_reid.json`.

**Note:** PostgreSQL 17 is now running as a background service on this machine (`brew services stop postgresql@17` to stop it). The `smartdetect_scale` database contains only synthetic benchmark vectors — no real biometric data.
