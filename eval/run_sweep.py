"""
eval/run_sweep.py
──────────────────
Runs eval/run_eval.py once per ablation config, in a SEPARATE PROCESS each
time (so the identity-config singleton, ByteTrack state, model singletons
and DB session registry all start cold — a config must not be able to leak
into the next one), then writes eval/results/summary.md.

Usage:
    python eval/run_sweep.py --adapter folder --data-root eval/data/demo_labeled
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # importable as `python eval/run_sweep.py`
CONFIGS = ["A_pre_hardening", "B_face_anchor", "C_plus_guard", "D_full"]


def fmt(v, nd=3, pct=False):
    if v is None:
        return "n/a"
    if isinstance(v, (int,)) and not pct:
        return str(v)
    if pct:
        return f"{100*v:.1f}%"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["folder", "chokepoint"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--results", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--allow-underpowered", action="store_true",
                    help=("Run the sweep even if the dataset fails the adequacy "
                          "gate. For pipeline smoke-testing ONLY — results from "
                          "an underpowered corpus must never be reported."))
    args = ap.parse_args()

    results_dir = Path(args.results).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset adequacy gate ───────────────────────────────────────────────
    from eval.validate_dataset import validate, format_report
    val = validate(args.adapter, args.data_root)
    print(format_report(val, args.adapter, str(Path(args.data_root).resolve())))
    (results_dir / "dataset_validation.json").write_text(json.dumps(val, indent=2))
    if not val["passed"]:
        if not args.allow_underpowered:
            sys.exit(
                "\nABLATION BLOCKED: dataset below minimum adequacy (see gates above).\n"
                "Add labelled data, or pass --allow-underpowered to smoke-test the\n"
                "pipeline — in which case the output is NOT a reportable result."
            )
        print("\n*** --allow-underpowered SET: continuing on an inadequate dataset. ***")
        print("*** Output is a pipeline smoke-test, NOT a reportable result.    ***\n")

    all_metrics = {}
    for name in CONFIGS:
        cfg_path = ROOT / "config" / "ablation" / f"{name}.json"
        if not cfg_path.is_file():
            sys.exit(f"missing ablation config: {cfg_path}")
        out_dir = results_dir / name
        print(f"\n{'='*70}\n=== {name}\n{'='*70}")
        cmd = [sys.executable, str(ROOT / "eval" / "run_eval.py"),
               "--config", str(cfg_path), "--adapter", args.adapter,
               "--data-root", args.data_root, "--out", str(out_dir)]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            sys.exit(f"run_eval failed for {name} (exit {r.returncode})")
        all_metrics[name] = json.loads((out_dir / "metrics.json").read_text())

    # ── summary.md ──────────────────────────────────────────────────────────
    from eval.scoring import fmt_ci
    first = all_metrics[CONFIGS[0]]
    lines = ["# Identity ablation results", ""]
    if not val["passed"]:
        vs = val["stats"]
        lines += [
            "> ## ⛔ NOT A REPORTABLE RESULT",
            "> The dataset failed the adequacy gate and this sweep ran with",
            "> `--allow-underpowered`. These numbers are a **pipeline smoke-test**.",
            f"> Corpus: {vs['n_identities']} identities, {vs['n_frames']} labelled frames, "
            f"{vs['n_cross_camera_transitions']} cross-camera transitions.",
            "> Required: 15 / 1000 / 10. See `dataset_validation.json`.",
            "> Confidence intervals below show how little these point estimates constrain.",
            "",
        ]
    lines += [
        f"Dataset: `{first['dataset']['adapter']}` at `{first['dataset']['root']}`  ",
        f"Frames per config: {first['runtime']['n_frames']}  ",
        f"Scored assignment records per config: {first['n_records']}  ",
        f"Machine: {first['runtime']['machine']}",
        "",
        f"Sighting dedup: {first.get('sighting_dedup_seconds', 0)} s of dataset time "
        f"(0 = one row per gate-passing frame)",
        "",
        "All metrics are computed against DATASET GROUND TRUTH over every processed",
        "person-frame. Nothing is read from `snapshots/` or any other system-produced",
        "artefact. Exact formulas: [METRICS.md](../METRICS.md)",
        "",
        "## Ground-truth bucket partition",
        "",
        "Every ground-truth person-frame falls in exactly one bucket; the three sum to 100%.",
        "",
        "All proportions carry 95% Wilson score intervals: `point [lo-hi]`.",
        "",
        "| Config | CORRECT | CONTAMINATED | UNASSIGNED | Σ | n |",
        "|---|---|---|---|---|---|",
    ]
    for name in CONFIGS:
        b = all_metrics[name]["buckets"]
        s = b["frac_correct"] + b["frac_contaminated"] + b["frac_unassigned"]
        lines.append(
            f"| {name} "
            f"| {b['n_correct']} · {fmt_ci(b['ci_correct'])} "
            f"| {b['n_contaminated']} · {fmt_ci(b['ci_contaminated'])} "
            f"| {b['n_unassigned']} · {fmt_ci(b['ci_unassigned'])} "
            f"| {fmt(s, pct=True)} "
            f"| {b['n_total']} |")

    lines += [
        "",
        "## Headline metrics",
        "",
        "**Precision and coverage are reported separately and must never be combined**",
        "into a single score: the design deliberately trades coverage for precision, and",
        "one number hides the axis under study.",
        "",
        "95% Wilson intervals. Fragmentation is a mean (not a proportion) so it",
        "carries no binomial interval; ID switches is a count.",
        "",
        "| Config | Identity precision ↑ | Coverage | Evidence precision ↑ | Ev. rows | Purity ↑ | Dupes ↓ | Frag. ↓ | ID sw. | Cross-cam ↑ | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in CONFIGS:
        m = all_metrics[name]
        ep = m["evidence_precision"]
        lines.append(
            f"| {name} "
            f"| {fmt_ci(m['identity_precision']['ci'])} "
            f"| {fmt_ci(m['identity_coverage']['ci'])} "
            f"| {fmt_ci(ep['ci'])} "
            f"| {ep['n_written']} ({ep['n_wrong']} wrong) "
            f"| {fmt_ci(m['identity_purity']['ci'])} "
            f"| {fmt(m['duplicate_identities']['value'])} "
            f"| {fmt(m['fragmentation']['value'])} "
            f"| {m['id_switches']['value']} "
            f"| {fmt_ci(m['cross_camera_reassoc'].get('ci'))} "
            f"| {m['runtime']['mean_frame_latency_ms']:.0f} |")

    lines += ["", "## Why coverage is not 100%", "",
              "| Config | detector miss | quality-gate refusal | total unassigned |",
              "|---|---|---|---|"]
    for name in CONFIGS:
        b = all_metrics[name]["buckets"]
        lines.append(f"| {name} | {b['n_unassigned_no_detection']} "
                     f"| {b['n_unassigned_detecting']} | {b['n_unassigned']} |")

    lines += ["", "## Harness caveats", ""]
    for note in first.get("harness_notes", []):
        lines.append(f"- {note}")
    lines += ["- Per-config runs are separate OS processes; no state is shared between configs.",
              "- `smartdetect.db` is never opened: each run uses its own `eval.db`.",
              ""]

    (results_dir / "summary.md").write_text("\n".join(lines))
    (results_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nwrote {results_dir/'summary.md'}")

    # Fail loudly if any config's partition is broken — a summary table that
    # does not sum to 100% must never be published.
    for name, m in all_metrics.items():
        b = m["buckets"]
        assert b["n_correct"] + b["n_contaminated"] + b["n_unassigned"] == b["n_total"], \
            f"{name}: bucket counts do not sum to total"
        s = b["frac_correct"] + b["frac_contaminated"] + b["frac_unassigned"]
        assert abs(s - 1.0) < 1e-9, f"{name}: bucket fractions sum to {s}"
    print("bucket partition verified: all configs sum to exactly 100%")


if __name__ == "__main__":
    main()
