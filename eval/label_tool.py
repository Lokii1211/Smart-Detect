"""
eval/label_tool.py
───────────────────
Interactive ground-truth labelling helper.

Steps through a video, runs the production person detector on each sampled
frame, and lets you tag every visible person with a ground-truth identity.
Writes the folder-adapter layout directly:

    <out>/<camera_id>/<person_id>/<frame>_<n>.jpg

Identities persist across runs and across cameras via <out>/_identities.json,
so labelling the same person on a second camera creates the cross-camera
transitions the ablation needs.

USAGE
    python eval/label_tool.py --video demo_videos/01_faces_walking.mp4 \
        --camera-id cam_hallway --out eval/data/mydataset --every 15

KEYS (window focused)
    0-9     assign that existing identity to the highlighted box
    n       new identity (type the name in the terminal)
    TAB     highlight next box in this frame
    SPACE   accept frame and advance
    s       skip frame entirely (labels nothing)
    u       undo last write
    b       back one sampled frame
    q       quit and save

The detector is only a labelling aid — you are the ground truth. If it misses
a person, that frame simply yields no label for them; if it boxes something
wrong, skip it. Never label a box you cannot personally identify: a guessed
label is worse than no label, because it silently becomes "truth".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def _gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


class Registry:
    """Identity names shared across cameras and runs."""

    def __init__(self, out_root: Path):
        self.path = out_root / "_identities.json"
        self.names: list[str] = []
        if self.path.is_file():
            try:
                self.names = json.loads(self.path.read_text())
            except Exception:
                self.names = []

    def add(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
            self.save()
        return self.names.index(name)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.names, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Label video frames for the eval harness")
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-id", required=True,
                    help="folder-adapter camera id, e.g. cam_entrance")
    ap.add_argument("--out", required=True, help="dataset root")
    ap.add_argument("--every", type=int, default=15,
                    help="sample every Nth frame (default 15)")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--pad", type=float, default=0.08,
                    help="fractional padding around each crop (default 0.08)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        sys.exit(f"video not found: {video}")
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not _gui_available():
        sys.exit(
            "OpenCV has no GUI backend on this install, so interactive labelling\n"
            "cannot run. Options:\n"
            "  * pip install opencv-python (not -headless) in this venv, or\n"
            "  * lay frames out by hand in the folder-adapter structure:\n"
            f"       {out_root}/<camera_id>/<person_id>/*.jpg\n"
            "    then validate with eval/validate_dataset.py"
        )

    from recognition.object_detector import ObjectDetector
    det = ObjectDetector()
    det.load_model()

    reg = Registry(out_root)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det.set_imgsz_for_resolution(w, h)

    print(f"{video.name}: {total} frames, sampling every {args.every}")
    print(f"writing to {out_root}/{args.camera_id}/<person_id>/")
    print("keys: 0-9 assign | n new | TAB next box | SPACE next frame | "
          "s skip | u undo | b back | q quit")

    indices = list(range(args.start, total, args.every))
    written: list[Path] = []
    pos = 0
    win = f"label [{args.camera_id}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= pos < len(indices):
        fidx = indices[pos]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            pos += 1
            continue

        boxes = [d["bbox"] for d in det.detect(frame) if d["label"] == "person"]
        boxes.sort(key=lambda b: -b[2] * b[3])
        sel = 0
        assigned: dict[int, str] = {}

        while True:
            vis = frame.copy()
            for i, (x, y, bw, bh) in enumerate(boxes):
                colour = (0, 215, 255) if i == sel else (80, 200, 80)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), colour, 2 if i == sel else 1)
                tag = assigned.get(i, f"[{i}] unlabelled")
                cv2.putText(vis, tag, (x, max(16, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            hud = [f"frame {fidx}/{total}  ({pos+1}/{len(indices)})  boxes={len(boxes)}",
                   f"written={len(written)}"]
            for i, nm in enumerate(reg.names[:10]):
                hud.append(f"  {i}: {nm}")
            for j, line in enumerate(hud):
                cv2.putText(vis, line, (8, 20 + 18 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, line, (8, 20 + 18 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF

            if k == ord("q"):
                pos = len(indices)
                break
            if k == ord("s"):
                assigned.clear()
                pos += 1
                break
            if k == ord("b"):
                pos = max(0, pos - 1)
                break
            if k == 9 and boxes:                      # TAB
                sel = (sel + 1) % len(boxes)
            elif k == ord("n"):
                cv2.destroyWindow(win)
                name = input("new identity id (e.g. person_03): ").strip()
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                if name:
                    reg.add(name)
                    if boxes:
                        assigned[sel] = name
            elif ord("0") <= k <= ord("9"):
                i = k - ord("0")
                if i < len(reg.names) and boxes:
                    assigned[sel] = reg.names[i]
            elif k == ord("u") and written:
                last = written.pop()
                try:
                    last.unlink()
                    print(f"undo: removed {last}")
                except OSError:
                    pass
            elif k == ord(" "):
                for i, name in assigned.items():
                    x, y, bw, bh = boxes[i]
                    px, py = int(bw * args.pad), int(bh * args.pad)
                    x0, y0 = max(0, x - px), max(0, y - py)
                    x1, y1 = min(frame.shape[1], x + bw + px), min(frame.shape[0], y + bh + py)
                    crop = frame[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    d = out_root / args.camera_id / name
                    d.mkdir(parents=True, exist_ok=True)
                    p = d / f"{fidx:08d}_{i}.jpg"
                    if cv2.imwrite(str(p), crop):
                        written.append(p)
                pos += 1
                break

    cap.release()
    cv2.destroyAllWindows()
    reg.save()

    print(f"\nwrote {len(written)} labelled crops to {out_root}/{args.camera_id}/")
    print("Validate the corpus with:")
    print(f"  python eval/validate_dataset.py --adapter folder --data-root {out_root}")


if __name__ == "__main__":
    main()
