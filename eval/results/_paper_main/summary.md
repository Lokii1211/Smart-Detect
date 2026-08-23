# Identity ablation results

Dataset: `chokepoint` at `/Users/lokii/Downloads/Smart-Detect-main/eval/data/chokepoint`  
Frames per config: 4200  
Scored assignment records per config: 1630  
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
| A_pre_hardening | 388 · 23.8% [21.8-25.9] | 1239 · 76.0% [73.9-78.0] | 3 · 0.2% [0.1-0.5] | 100.0% | 1630 |
| B_face_anchor | 754 · 46.3% [43.8-48.7] | 855 · 52.5% [50.0-54.9] | 21 · 1.3% [0.8-2.0] | 100.0% | 1630 |
| C_plus_guard | 1602 · 98.3% [97.5-98.8] | 1 · 0.1% [0.0-0.3] | 27 · 1.7% [1.1-2.4] | 100.0% | 1630 |
| D_full | 1602 · 98.3% [97.5-98.8] | 1 · 0.1% [0.0-0.3] | 27 · 1.7% [1.1-2.4] | 100.0% | 1630 |

## Headline metrics

**Precision and coverage are reported separately and must never be combined**
into a single score: the design deliberately trades coverage for precision, and
one number hides the axis under study.

95% Wilson intervals. Fragmentation is a mean (not a proportion) so it
carries no binomial interval; ID switches is a count.

| Config | Identity precision ↑ | Coverage | Evidence precision ↑ | Ev. rows | Purity ↑ | Dupes ↓ | Frag. ↓ | ID sw. | Cross-cam ↑ | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| A_pre_hardening | 23.8% [21.8-26.0] | 99.8% [99.5-99.9] | 23.8% [21.8-26.0] | 1627 (1239 wrong) | 0.0% [0.0-39.0] | 0.500 | 2.929 | 25 | 10.7% [3.7-27.2] | 65 |
| B_face_anchor | 46.9% [44.4-49.3] | 98.7% [98.0-99.2] | 46.9% [44.4-49.3] | 1609 (855 wrong) | 43.8% [23.1-66.8] | 0.455 | 2.857 | 25 | 14.3% [5.7-31.5] | 62 |
| C_plus_guard | 99.9% [99.6-100.0] | 98.3% [97.6-98.9] | 99.9% [99.6-100.0] | 1603 (1 wrong) | 97.2% [85.8-99.5] | 1.571 | 2.643 | 32 | 42.9% [26.5-60.9] | 64 |
| D_full | 99.9% [99.6-100.0] | 98.3% [97.6-98.9] | 100.0% [99.7-100.0] | 1313 (0 wrong) | 97.2% [85.8-99.5] | 1.571 | 2.643 | 32 | 42.9% [26.5-60.9] | 64 |

## Why coverage is not 100%

| Config | detector miss | quality-gate refusal | total unassigned |
|---|---|---|---|
| A_pre_hardening | 0 | 3 | 3 |
| B_face_anchor | 0 | 21 | 21 |
| C_plus_guard | 0 | 27 | 27 |
| D_full | 0 | 27 | 27 |

## Harness caveats

- head-zoom second pass not exercised offline (needs capture state from LiveStream.start()); face recall here is a lower bound vs production
- Per-config runs are separate OS processes; no state is shared between configs.
- `smartdetect.db` is never opened: each run uses its own `eval.db`.
