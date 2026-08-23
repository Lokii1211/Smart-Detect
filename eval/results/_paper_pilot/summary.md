# Identity ablation results

Dataset: `chokepoint` at `/Users/lokii/Downloads/Smart-Detect-main/eval/data/chokepoint`  
Frames per config: 750  
Scored assignment records per config: 41  
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
| A_pre_hardening | 40 · 97.6% [87.4-99.6] | 0 · 0.0% [0.0-8.6] | 1 · 2.4% [0.4-12.6] | 100.0% | 41 |
| B_face_anchor | 40 · 97.6% [87.4-99.6] | 0 · 0.0% [0.0-8.6] | 1 · 2.4% [0.4-12.6] | 100.0% | 41 |
| C_plus_guard | 40 · 97.6% [87.4-99.6] | 0 · 0.0% [0.0-8.6] | 1 · 2.4% [0.4-12.6] | 100.0% | 41 |
| D_full | 40 · 97.6% [87.4-99.6] | 0 · 0.0% [0.0-8.6] | 1 · 2.4% [0.4-12.6] | 100.0% | 41 |

## Headline metrics

**Precision and coverage are reported separately and must never be combined**
into a single score: the design deliberately trades coverage for precision, and
one number hides the axis under study.

95% Wilson intervals. Fragmentation is a mean (not a proportion) so it
carries no binomial interval; ID switches is a count.

| Config | Identity precision ↑ | Coverage | Evidence precision ↑ | Ev. rows | Purity ↑ | Dupes ↓ | Frag. ↓ | ID sw. | Cross-cam ↑ | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| A_pre_hardening | 100.0% [91.2-100.0] | 97.6% [87.4-99.6] | 100.0% [91.2-100.0] | 40 (0 wrong) | 100.0% [34.2-100.0] | 1.000 | 2.000 | 0 | 0.0% [0.0-79.3] | 132 |
| B_face_anchor | 100.0% [91.2-100.0] | 97.6% [87.4-99.6] | 100.0% [91.2-100.0] | 40 (0 wrong) | 100.0% [34.2-100.0] | 1.000 | 2.000 | 0 | 0.0% [0.0-79.3] | 131 |
| C_plus_guard | 100.0% [91.2-100.0] | 97.6% [87.4-99.6] | 100.0% [91.2-100.0] | 40 (0 wrong) | 100.0% [34.2-100.0] | 1.000 | 2.000 | 0 | 0.0% [0.0-79.3] | 135 |
| D_full | 100.0% [91.2-100.0] | 97.6% [87.4-99.6] | 100.0% [88.3-100.0] | 29 (0 wrong) | 100.0% [34.2-100.0] | 1.000 | 2.000 | 0 | 0.0% [0.0-79.3] | 131 |

## Why coverage is not 100%

| Config | detector miss | quality-gate refusal | total unassigned |
|---|---|---|---|
| A_pre_hardening | 0 | 1 | 1 |
| B_face_anchor | 0 | 1 | 1 |
| C_plus_guard | 0 | 1 | 1 |
| D_full | 0 | 1 | 1 |

## Harness caveats

- head-zoom second pass not exercised offline (needs capture state from LiveStream.start()); face recall here is a lower bound vs production
- Per-config runs are separate OS processes; no state is shared between configs.
- `smartdetect.db` is never opened: each run uses its own `eval.db`.
