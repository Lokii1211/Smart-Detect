# Identity ablation results

Dataset: `chokepoint` at `/Users/lokii/Downloads/Smart-Detect-main/eval/data/chokepoint`  
Frame set: **full corpus, no cap**  
Scored corpus: 25 identities · 2908 labelled frames · 50 cross-camera transitions (gate: 15/1000/10)  
Frames per config: 6876  
Scored assignment records per config: 2908  
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
| A_pre_hardening | 449 · 15.4% [14.2-16.8] | 2454 · 84.4% [83.0-85.7] | 5 · 0.2% [0.1-0.4] | 100.0% | 2908 |
| B_face_anchor | 1542 · 53.0% [51.2-54.8] | 1324 · 45.5% [43.7-47.3] | 42 · 1.4% [1.1-1.9] | 100.0% | 2908 |
| C_plus_guard | 2859 · 98.3% [97.8-98.7] | 9 · 0.3% [0.2-0.6] | 40 · 1.4% [1.0-1.9] | 100.0% | 2908 |
| D_full | 2859 · 98.3% [97.8-98.7] | 9 · 0.3% [0.2-0.6] | 40 · 1.4% [1.0-1.9] | 100.0% | 2908 |

## Headline metrics

**Precision and coverage are reported separately and must never be combined**
into a single score: the design deliberately trades coverage for precision, and
one number hides the axis under study.

95% Wilson intervals. Fragmentation is a mean (not a proportion) so it
carries no binomial interval; ID switches is a count.

| Config | Identity precision ↑ | Coverage | Evidence precision ↑ | Ev. rows | Purity ↑ | Dupes ↓ | Frag. ↓ | ID sw. | Cross-cam ↑ | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| A_pre_hardening | 15.5% [14.2-16.8] | 99.8% [99.6-99.9] | 15.5% [14.2-16.8] | 2903 (2454 wrong) | 0.0% [0.0-35.4] | 0.400 | 2.960 | 47 | 8.0% [3.2-18.8] | 56 |
| B_face_anchor | 53.8% [52.0-55.6] | 98.6% [98.1-98.9] | 53.8% [52.0-55.6] | 2866 (1324 wrong) | 48.3% [31.4-65.6] | 0.381 | 2.560 | 44 | 28.0% [17.5-41.7] | 59 |
| C_plus_guard | 99.7% [99.4-99.8] | 98.6% [98.1-99.0] | 99.7% [99.4-99.8] | 2868 (9 wrong) | 93.2% [83.8-97.3] | 1.360 | 2.520 | 57 | 50.0% [36.6-63.4] | 61 |
| D_full | 99.7% [99.4-99.8] | 98.6% [98.1-99.0] | 100.0% [99.8-100.0] | 2402 (0 wrong) | 93.2% [83.8-97.3] | 1.360 | 2.520 | 57 | 50.0% [36.6-63.4] | 62 |

## Why coverage is not 100%

| Config | detector miss | quality-gate refusal | total unassigned |
|---|---|---|---|
| A_pre_hardening | 2 | 3 | 5 |
| B_face_anchor | 2 | 40 | 42 |
| C_plus_guard | 2 | 38 | 40 |
| D_full | 2 | 38 | 40 |

## Harness caveats

- head-zoom second pass not exercised offline (needs capture state from LiveStream.start()); face recall here is a lower bound vs production
- Per-config runs are separate OS processes; no state is shared between configs.
- `smartdetect.db` is never opened: each run uses its own `eval.db`.
