# SmartDetect — Independent Engineering Review

**Reviewer role:** Senior AI / Computer Vision Engineer
**Date:** 2026-07-31
**Codebase:** ~10.3 kLOC Python + ~4.1 kLOC React, commit `b784b1a` + uncommitted eval/config work
**Basis:** direct code audit and measurements taken on this machine (Apple M5, 16 GB). Every number below was produced by running the system, not estimated.

---

## 1. Verdict up front

SmartDetect is a **solid single-camera demonstrator with unusually disciplined identity-arbitration logic**, and it is **not yet a deployable multi-camera surveillance system**. The gap is not polish — it is throughput architecture, evaluation rigour, security, and data governance.

The strongest engineering in this project is the identity decision layer: the face-anchored matching, the ID-switch guard, and the face-confirmed evidence gate are genuinely well-reasoned, and each is now parameterised and independently verified. The weakest areas are that **no model was trained or fine-tuned**, **the headline accuracy result is measured with a circular method**, and **13 endpoints of a face-recognition system are unauthenticated**.

| Dimension | State | Confidence |
|---|---|---|
| Identity arbitration logic | Strong, verified | High — flag-by-flag proof |
| Single-camera / single-video pipeline | Works end to end | High — run repeatedly |
| Multi-camera concurrency | Hard cap of 4, untested beyond that | High |
| Crowd / high-density performance | Degrades to counting-only by design | High — measured |
| ML training / fine-tuning | **None performed** | Certain |
| Evaluation rigour | Weak; primary metric is circular | High |
| Security posture | **Not deployable as-is** | Certain |
| Production DB path (Postgres/pgvector) | Written, never executed | Certain |

---

## 2. What is genuinely built and verified

Claims here are ones I personally reproduced this session.

### 2.1 Working pipeline
- **Detection → tracking → face recognition → re-ID → identity assignment**, end to end, on webcam, uploaded video, and RTSP-capable sources.
- YOLOv8n person detection: **77.5 fps** on 768×576 (M5 CPU, detection only, 795 frames).
- Video upload → analysis → EOF → auto-offline, verified with correct fps/frame-count/duration parsing.
- 4 simultaneous video streams at full native playback fps.
- React dashboard: live CCTV wall, People management, realtime Photo Search, search-by-ID, duplicate-merge UI.

### 2.2 Identity arbitration (the real intellectual contribution)
Five mechanisms, all now externalised to `config/identity_config.py` and **each independently proven to gate control flow** using real database records (`scripts/verify_flags.py`):

| Mechanism | Verified effect |
|---|---|
| Face-anchored matching | Stranger's face blocks colour re-association (`SDT-0001` → `Detecting...`) |
| Colour fallback switch | Toggles Method 2 on/off |
| Re-ID fallback switch | Toggles Method 3 on/off |
| ID-switch guard | Cache dropped vs retained on 2 contradictions |
| Evidence gating | 0 vs 1 sighting rows written; snapshot suppressed vs written |

Measured ablation (A→C): **contamination 41.7% → 2.8%**, traced frame-by-frame. This is a real, explainable result.

### 2.3 Engineering hygiene added
- Config-driven parameters with typo-protecting loader and startup logging.
- Offline evaluation harness that drives production code paths (no logic duplication).
- Documented metric formulas (`eval/METRICS.md`).
- Purity regression check (`scripts/purity_eval.py`).

---

## 3. What was **not** done — read this before any paper or pitch

### 3.1 No model was trained. At all.
There is **zero training code** in the repository — no optimiser, no loss, no backward pass, no epochs. Verified by exhaustive grep.

Every model is off-the-shelf pretrained:

| Model | Origin | Trained by |
|---|---|---|
| YOLOv8n | Ultralytics COCO checkpoint | Ultralytics |
| InsightFace `buffalo_l` (ArcFace) | InsightFace release v0.7 | InsightFace / DeepInsight |
| OSNet x1.0 | deep-person-reid model zoo (MSMT17+Duke+CUHK03) | Kaiyang Zhou et al. |

**This is a legitimate and normal engineering choice** — pretrained face embeddings outperform anything trainable on a small custom dataset. But it must be described accurately. The contribution of this project is **decision-layer engineering and systems integration**, not model training or representation learning.

> ⚠️ If a paper, report, or CV describes these models as "trained" or "fine-tuned", that is a factual misstatement that a reviewer will catch immediately. Describe it as: *"a multi-stage identity arbitration system built on pretrained detection, face-recognition and re-ID backbones."*

### 3.2 The headline accuracy result is circular
`scripts/purity_eval.py` reports **"0/15 codes contain merged identities."** That result is substantially weaker than it appears:

1. Snapshots are only written when `_evidence_gate_ok()` returns True.
2. That gate requires a face matching the assigned code at ≥ 0.45 cosine similarity.
3. `purity_eval.py` then measures whether the faces in those snapshots are mutually similar (threshold 0.20).

**The sample is filtered by the property being measured.** Frames that would have proven contamination are the exact frames the gate refuses to save. The test is close to guaranteed to pass by construction.

Partial mitigations: registration photos (`registered.jpg`) are *not* evidence-gated, so cross-registration comparisons retain some independence; and the ablation harness did surface a genuinely contaminated frame under config C. But as a primary accuracy claim it does not hold up.

**Fix required:** score against *external* ground truth (labelled frames), never against system-selected evidence. The `eval/` harness is the right foundation; it needs a real dataset.

### 3.3 Evaluation is far too small
Current labelled set: **2 identities, 45 frames, 1 cross-camera pair.** Purity moves in 50-point increments. Cross-camera accuracy rests on a single observation. No statistical power whatsoever.

### 3.4 Untested / incomplete
| Item | Status |
|---|---|
| PostgreSQL + pgvector path | Code written, **never executed** |
| Docker Compose | Predates video upload; stale |
| Unit tests | **None.** The four `scripts/*test*.py` are integration smoke scripts |
| `scripts/accuracy_test.py` | Uses **synthetic random embeddings** — its "100% accuracy" measures NumPy's RNG, not face recognition. Not a valid benchmark |
| `crowd` / `camera_offline` alerts | UI and schema exist; backend never emits them |
| Multi-feature fusion (Method 4) | Documented in docstring, **never implemented** |
| `scripts/seed_stations.py` | Stale — posts to `/stations`, API is `/locations` |
| DeepSORT references in PROJECT_DOCUMENTATION.md | Stale — system uses ByteTrack |

---

## 4. Critical findings

### 4.1 🔴 SECURITY — not deployable as-is
**13 endpoints require no authentication**, including:

```
GET  /persons                 → every tracked person + photo paths
GET  /persons/{code}          → full appearance history
GET  /camera/stream/{id}      → live annotated video
POST /search/by-photo         → face search against the entire database
GET  /camera/detections/recent
```

Anyone who can reach the port can enumerate every person the system has ever seen, watch live cameras, and run face searches. For a biometric surveillance system this is the single most serious defect in the project.

Compounding issues:
- Default credentials hardcoded as fallbacks (`admin/smartAdmin2024`, `operator/smartOp2024`).
- JWT secret has a hardcoded development fallback.
- Dashboard auto-logs-in with operator credentials embedded in client-side JS.
- Rate limiting applied to `/search/by-photo` only.
- No HTTPS/TLS anywhere in the stack.

### 4.2 🔴 Legal / privacy — unaddressed
This is a **biometric identification system**. It stores face embeddings and photographs of identifiable people. There is currently:
- No retention or deletion policy (snapshots and embeddings accumulate indefinitely).
- No consent or notice mechanism.
- No audit log of who searched for whom.
- No data-subject access/erasure path.

Under GDPR (Art. 9 — biometric data), Illinois BIPA, and India's DPDP Act, deploying this against real people without those controls creates genuine legal exposure. **This needs a decision before any pilot, not after.**

### 4.3 🟠 Duplicate-identity rate is the accepted design trade-off
The system deliberately splits rather than merges when uncertain. Observed: the same woman registered twice (frontal vs head-bowed). Merge tooling exists but is manual. Expect operator workload proportional to pose variety.

---

## 5. Multi-camera and high-crowd readiness — the honest analysis

This is where the project is furthest from the goal, so I want to be precise rather than encouraging.

### 5.1 Measured throughput

| Stage | Measured |
|---|---|
| YOLO detection only, 768×576 | 77.5 fps |
| **Full identity pipeline (offline, synchronous)** | **~4.4 fps (≈225 ms/frame)** |
| Live stream, in-app analysis rate | ~1.8 analysed fps (playback 11.5 fps) |
| Concurrent streams supported | **4** (`MAX_STREAMS`, `backend/main.py:113`) |

The 17× gap between raw detection and full pipeline is InsightFace full-frame scanning plus **per-person OSNet re-ID and per-person arbitration**.

### 5.2 Why crowds are the hard case
Cost scales with **people per frame**, not just resolution:

- Store-aisle clip: avg 3.2, peak 5 people → identity pipeline cost roughly 3–5× the single-person case.
- Measured on real footage: overhead/distant crowd → **0 identities minted** (faces below the 48 px / 0.60 quality gate). Tracked and counted correctly, but no IDs.
- Eye-level crowd (4 people, same frame) → **4 identities, correctly separated**.

**The determining factor is face visibility, not headcount.** This is by design and it is the right design — but it means:

> For overhead or wide-angle crowd cameras, SmartDetect is a **people counter and tracker**, not an identification system. No amount of tuning changes that without a different sensing approach.

### 5.3 What "live multi-camera at scale" actually requires

Current architecture cannot reach it by parameter tuning. Required changes, in dependency order:

**Tier 1 — throughput (mandatory)**
1. **GPU inference.** Move off CPU ONNXRuntime. Either CoreML/ANE via `onnxruntime-silicon` on Apple silicon, or CUDA/TensorRT on an NVIDIA box. Expect 5–20× on the face stage. *Without this, nothing else matters.*
2. **Batch face inference** across cameras instead of per-frame per-camera calls.
3. **Decouple re-ID.** OSNet currently runs per person per cycle. Run it only on face-confirmed registrations and on re-association candidates.
4. **Adaptive frame skipping** — analyse every Nth frame, interpolate boxes between. Identity does not need 30 Hz.

**Tier 2 — architecture**
5. **Process-per-camera or worker pool**, not threads in one Python process. The GIL is a hard ceiling; `MAX_STREAMS=4` is a symptom, not a cause.
6. **Message queue** (Redis/RabbitMQ) between capture and analysis workers; capture nodes at the edge, GPU inference centralised.
7. **PostgreSQL + pgvector** — SQLite is single-writer and will thrash with N concurrent camera writers. This path is already written and needs to be *executed and load-tested*.
8. **Vector index (HNSW/IVFFlat).** Identity matching is currently a **linear scan over all persons in Python** (`_find_person_python`). At 10 k identities this is the bottleneck; at 100 k it is unusable.

**Tier 3 — cross-camera identity**
9. **Global identity resolution service.** Today each `LiveStream` owns its own track state; cross-camera association happens only incidentally via the shared face gallery. A real multi-camera system needs an explicit re-identification service with camera topology and transition-time priors.
10. **Camera calibration / zone graph** so "person left camera A heading east" constrains candidate matches on camera B.

**Tier 4 — crowd-specific**
11. **Higher-resolution cameras + optical zoom on chokepoints** rather than wide overhead views, if identification (not counting) is the goal.
12. **Face-pose quality gate at registration** — reject extreme yaw/pitch before minting a code (would have prevented the observed duplicate). Landmarks are already available from InsightFace.
13. **Tracklet-level identity voting** instead of per-frame assignment — decide identity once per tracklet using the best K frames, which is both more accurate and cheaper.

### 5.4 Realistic capacity estimate
On current architecture, this machine: **4 cameras, ~2 analysed fps each, eye-level, low density.**
With Tier 1 + Tier 2 on a single mid-range GPU: **plausibly 12–20 cameras at 5 fps analysis.** That estimate is *unvalidated* — treat it as a hypothesis to test, not a specification.

---

## 6. Prioritised roadmap

**P0 — before any real-world deployment**
1. Authentication on all data endpoints; remove hardcoded credentials and JWT fallback; TLS.
2. Data-retention policy, deletion path, audit logging; legal review for the target jurisdiction.
3. Replace the circular purity metric with external-ground-truth evaluation.

**P1 — before claiming accuracy numbers**
4. Acquire a real labelled dataset (ChokePoint adapter is written and waiting; or label ~30 min of representative footage).
5. Run the 4-config ablation at scale; promote **evidence precision** to a first-class metric.
6. Add a pytest unit suite around the identity arbitration logic; delete or rewrite `accuracy_test.py`.

**P2 — scale**
7. GPU inference → measure → then process-per-camera → then Postgres/pgvector + HNSW.
8. Load-test at 8, 16, 32 cameras and publish the curve.

**P3 — accuracy/quality**
9. Face-pose gate at registration (reduces duplicates).
10. Tracklet-level identity voting.
11. Auto-merge suggestions with confidence banding (keep human confirmation).

**P4 — hygiene**
12. Reconcile PROJECT_DOCUMENTATION.md with reality (DeepSORT → ByteTrack, threading model).
13. Remove or implement: Method 4 fusion, `crowd`/`camera_offline` alerts, `seed_stations.py`.

---

## 7. Questions I need answered to sharpen this

1. **Deployment target** — how many cameras, what resolution, indoor/outdoor, and are they at face height or overhead? This single answer determines whether identification is achievable at all.
2. **Purpose** — is the goal *identification* (who is this person) or *analytics* (how many, where, how long)? The current design is over-engineered for the latter and under-resourced for the former at scale.
3. **Is this academic or commercial?** If academic (the IEEE paper mentioned earlier), the circular-metric issue is the top priority. If commercial, security and legal are.
4. **Hardware budget** — is a GPU available? Everything in Tier 1 assumes one.
5. **Who are the subjects?** Consenting staff, or members of the public? This changes the legal analysis fundamentally.

---

## 8. Bottom line

**What you have:** a well-architected single-camera identity system with a genuinely thoughtful arbitration layer, a working dashboard, and — as of this week — proper configuration management and an evaluation harness. The identity-hardening work is real and the ablation showing 41.7% → 2.8% contamination is a legitimate result.

**What you do not have:** any trained model, a trustworthy accuracy number, authentication, a legal basis for deployment, or an architecture that scales past 4 cameras.

**The most valuable next step** is not a feature. It is acquiring real labelled multi-camera footage and re-running the ablation against it. Everything else — scale, tuning, publication — depends on having an accuracy number you can defend.
