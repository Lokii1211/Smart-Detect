# Archived evaluation results — SUPERSEDED, DO NOT CITE

> **Nothing in this directory is a current result.** Every file here was
> produced before a later change invalidated it. They are retained only so the
> project's measurement history is auditable — specifically, so that the claim
> "the ablation was re-run after each performance change" can be checked
> against artefacts rather than taken on trust.
>
> For current numbers use `eval/results/_paper_main/` (ChokePoint) and
> regenerate with `eval/run_sweep.py`. Formulas: [`eval/METRICS.md`](../../METRICS.md).

Archived **2026-08-05**.

---

## What is here

All seven artefacts come from **one sweep**, run 2026-08-01 13:30–13:31:

| Artefact | Contents |
|---|---|
| `summary.md` | The results table for that sweep |
| `all_metrics.json` | Merged metrics for all four configurations |
| `dataset_validation.json` | The adequacy-gate report for that run |
| `A_pre_hardening/`, `B_face_anchor/`, `C_plus_guard/`, `D_full/` | Per-config `metrics.json`, `assignments.json`, `eval.db`, `workdir/` |

**Provenance of that run**

- Dataset: `folder` adapter on `eval/data/demo_labeled`
- Scored: **45 person-frames, 2 identities, 1 cross-camera transition**
- Gate verdict: **FAILED** — it ran under `--allow-underpowered`
- Machine: Apple M5, ONNXRuntime **CPUExecutionProvider** (no CoreML)
- Mean latency: 228.6 ms/frame

`summary.md` carries its own "⛔ NOT A REPORTABLE RESULT" banner. That banner
was correct when written and remains correct.

---

## What invalidated it

Four independent changes, all landing after the run:

### 1. CoreML execution provider — 2026-08-02 09:29
`recognition/face_recognizer.py` began preferring `CoreMLExecutionProvider`.
The archived latency of **228.6 ms/frame** was measured on CPU-only inference
and is roughly **2.4× the current cost**. Any timing here is obsolete.

### 2. Lazy OSNet re-ID — 2026-08-02 22:50
`recognition/smart_identifier.py` moved the body-re-ID forward pass behind a
lazy helper, computed at most once and only when a consumer needs the vector.
Further reduces per-frame cost; identity semantics unchanged.

### 3. `duplicate_identities` metric added — 2026-08-02 22:54
`eval/scoring.py` gained a metric that separates genuine over-splitting from
contamination. **These files predate it** — the key is absent from every
`metrics.json` in this directory. Any duplicate-rate claim sourced here is not
merely stale, it is unmeasured.

### 4. Registration pose gate — 2026-08-02 23:06
`recognition/face_pose.py` and `config/identity_config.py` added a pose gate at
enrolment, plus a starvation valve. This changes which frames mint identities,
so coverage and duplicate counts here no longer describe current behaviour.

### 5. Adequacy gate corrected for truncation — 2026-08-04/05
`eval/validate_dataset.py` and `eval/run_sweep.py` were changed so the gate
evaluates the **scored** frame set rather than the full corpus. The
`dataset_validation.json` here predates that fix and has no `max_frames` field.

---

## Who cited it

At archive time: **nothing.** No document, script, or test referenced
`eval/results/summary.md`. `eval/run_sweep.py` mentions the *filename* because
it writes one, not because it reads this copy.

The IEEE manuscript in `paper/` draws its numbers from
`eval/results/_paper_main/`, not from here.

---

## Known defect in `_paper_main` as well

`_paper_main` is newer and is the source for the manuscript, but it is **not
itself fully clean**: it was run with `--max-frames 1400`, and the scored
subset contains **14 identities**, below the 15-identity gate minimum. The
corpus on disk has 25; truncation removed 11 people. The gate did not catch
this at the time because it validated the full corpus before truncation — the
defect fixed in change 5 above.

A full-corpus re-run is required before any table is reported.
