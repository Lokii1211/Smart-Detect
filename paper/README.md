# SmartDetect — IEEE conference paper

**SmartDetect: Face-Anchored Identity Arbitration for Auditable Cross-Camera
Person Re-Identification**

Krishiv Kameshwar B. S. · Kishan S. G. · Lokeshkumar D.
Supervisor: Ms. K. Sinduja

Dept. of Information Technology (B.Tech), Sri Krishna College of Technology,
Coimbatore, Tamil Nadu, India

## Deliverables

| File | What it is |
|---|---|
| `SmartDetect_IEEE_Paper.docx` | Submission manuscript. IEEE two-column, Times New Roman, ~11 500 words, 8 figures, 6 tables. |
| `SmartDetect_IEEE_Paper.html` | Same paper as a web page (published artifact). |
| `figures/` | All figures at 300 dpi, plus the raw annotated run frames. |

## Where the numbers come from

Every figure and table cell is computed from `eval/results/_paper_main/`,
the four-configuration ablation over ChokePoint P1E_S1 (3 cameras, 25
identities, 1630 scored person-frames per configuration). Nothing is
hand-entered. Regenerate with:

```bash
python eval/run_sweep.py --adapter chokepoint \
    --data-root eval/data/chokepoint --max-frames 1400 \
    --results eval/results/_paper_main
```

## Regenerating the figures

Run from the repository root with the project venv:

```bash
python paper/make_frames.py        # annotated frames from the live pipeline
python paper/compose_panels.py     # -> fig_operation.png  (Fig. 1)
python paper/make_pipeline_fig.py  # -> fig_pipeline.png   (Fig. 2)
python paper/make_charts.py        # -> Figs. 3-8 from metrics.json
python paper/build_docx.py         # -> the .docx
```

`make_frames.py` drives the real production code path (YOLOv8 → ByteTrack →
InsightFace → `SmartIdentifier`) over ChokePoint and draws the system's own
per-frame output. Every box, identifier and gate verdict in Fig. 1 is a real
decision, not an illustration.

Scripts write their outputs to a scratch directory; adjust the `OUT` path at
the top of each if you want them written here instead.

## Figures

| Fig. | Source | Content |
|---|---|---|
| 1 | real run | System in operation — quality gate, two identities held apart, enrolment, identity retained without a face |
| 2 | diagram | Per-frame decision path; stages 5–8 are the identity-critical ones |
| 3 | metrics | Outcome partition (CORRECT / CONTAMINATED / UNASSIGNED) |
| 4 | assignments | Identity-collapse matrix, config A vs D — the central result |
| 5 | metrics | Split-over-merge asymmetry |
| 6 | metrics | Precision, coverage and purity across the ablation |
| 7 | metrics | Evidence precision vs yield (the only pair separating C from D) |
| 8 | metrics | Per-frame latency, CPU-only |

## Before submitting

- Replace the placeholder venue line (top of page 1) with the real conference.
- Add a contact e-mail per author if the target template requires one — the
  author block is laid out to take a fifth line without reflowing.
- Confirm the reference list against the target format (currently IEEE style).
