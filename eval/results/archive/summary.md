<!-- ══════════════════════════════════════════════════════════════════
     ARCHIVED 2026-08-05 — SUPERSEDED, DO NOT CITE.
     Invalidated by: CoreML provider (08-02 09:29), lazy OSNet re-ID
     (08-02 22:50), the duplicate_identities metric (08-02 22:54, absent
     from this file), the registration pose gate (08-02 23:06), and the
     truncation-aware adequacy gate (08-04/05).
     Scored 45 person-frames over 2 identities and FAILED the gate; ran
     under --allow-underpowered. Latency here is CPU-only, ~2.4x current.
     Current results: eval/results/_paper_main/   See ./README.md
     ══════════════════════════════════════════════════════════════════ -->

# Identity ablation results

> ## ⛔ NOT A REPORTABLE RESULT
> The dataset failed the adequacy gate and this sweep ran with
> `--allow-underpowered`. These numbers are a **pipeline smoke-test**.
> Corpus: 2 identities, 45 labelled frames, 1 cross-camera transitions.
> Required: 15 / 1000 / 10. See `dataset_validation.json`.
> Confidence intervals below show how little these point estimates constrain.

Dataset: `folder` at `/Users/lokii/Downloads/Smart-Detect-main/eval/data/demo_labeled`  
Frames per config: 45  
Scored assignment records per config: 45  
Machine: Apple M5 (CPU inference, ONNXRuntime CPUExecutionProvider)

Sighting dedup: 0.0 s of dataset time (0 = one row per gate-passing frame)

All metrics are computed against DATASET GROUND TRUTH over every processed
person-frame. Nothing is read from `snapshots/` or any other system-produced
artefact. Exact formulas: [METRICS.md](../METRICS.md)

## Ground-truth bucket partition

Every ground-truth person-frame falls in exactly one bucket; the three sum to 100%.

All proportions carry 95% Wilson score intervals: `point [lo-hi]`.

| Config | CORRECT | CONTAMINATED | UNASSIGNED | Σ | n |
|---|---|---|---|---|---|
| A_pre_hardening | 21 · 46.7% [32.9-60.9] | 15 · 33.3% [21.4-47.9] | 9 · 20.0% [10.9-33.8] | 100.0% | 45 |
| B_face_anchor | 21 · 46.7% [32.9-60.9] | 15 · 33.3% [21.4-47.9] | 9 · 20.0% [10.9-33.8] | 100.0% | 45 |
| C_plus_guard | 35 · 77.8% [63.7-87.5] | 1 · 2.2% [0.4-11.6] | 9 · 20.0% [10.9-33.8] | 100.0% | 45 |
| D_full | 35 · 77.8% [63.7-87.5] | 1 · 2.2% [0.4-11.6] | 9 · 20.0% [10.9-33.8] | 100.0% | 45 |

## Headline metrics

**Precision and coverage are reported separately and must never be combined**
into a single score: the design deliberately trades coverage for precision, and
one number hides the axis under study.

95% Wilson intervals. Fragmentation is a mean (not a proportion) so it
carries no binomial interval; ID switches is a count.

| Config | Identity precision ↑ | Coverage | Evidence precision ↑ | Ev. rows | Purity ↑ | Frag. ↓ | ID sw. | Cross-cam ↑ | ms/frame |
|---|---|---|---|---|---|---|---|---|---|
| A_pre_hardening | 58.3% [42.2-72.9] | 80.0% [66.2-89.1] | 58.3% [42.2-72.9] | 36 (15 wrong) | 0.0% [0.0-79.3] | 1.000 | 0 | 100.0% [20.7-100.0] | 224 |
| B_face_anchor | 58.3% [42.2-72.9] | 80.0% [66.2-89.1] | 58.3% [42.2-72.9] | 36 (15 wrong) | 0.0% [0.0-79.3] | 1.000 | 0 | 100.0% [20.7-100.0] | 222 |
| C_plus_guard | 97.2% [85.8-99.5] | 80.0% [66.2-89.1] | 97.2% [85.8-99.5] | 36 (1 wrong) | 50.0% [9.5-90.5] | 1.500 | 2 | 100.0% [20.7-100.0] | 228 |
| D_full | 97.2% [85.8-99.5] | 80.0% [66.2-89.1] | 100.0% [88.6-100.0] | 30 (0 wrong) | 50.0% [9.5-90.5] | 1.500 | 2 | 100.0% [20.7-100.0] | 229 |

## Why coverage is not 100%

| Config | detector miss | quality-gate refusal | total unassigned |
|---|---|---|---|
| A_pre_hardening | 6 | 3 | 9 |
| B_face_anchor | 6 | 3 | 9 |
| C_plus_guard | 6 | 3 | 9 |
| D_full | 6 | 3 | 9 |

## Harness caveats

- head-zoom second pass not exercised offline (needs capture state from LiveStream.start()); face recall here is a lower bound vs production
- Per-config runs are separate OS processes; no state is shared between configs.
- `smartdetect.db` is never opened: each run uses its own `eval.db`.
