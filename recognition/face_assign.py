"""
recognition/face_assign.py
───────────────────────────
Optimal face-to-person assignment.

THE PROBLEM
───────────
The shipped rule is greedy and local: for each person box, take the first
detected face whose centroid falls inside the upper 60% of that box. In a dense
scene two person boxes overlap, one face centroid lies inside both, and whichever
box is visited first claims it. The face is real and clears the quality gate, so
every downstream mechanism — the face veto, the ID-switch guard, the evidence
gate — sees a legitimate face and has no reason to object. The wrong person is
identified with full confidence.

THE APPROACH
────────────
Score every (face, person box) pair on four geometric terms, then solve the
whole frame at once as a linear assignment problem, one-to-one. A face can no
longer be claimed by the first box that happens to contain it; it goes to the
box that explains it best given every other face and box in the frame.

    containment   fraction of the face box inside the person box — dominant
    vertical      face centre's height within the box; heads are near the top
    scale         face height as a fraction of person height, against an
                  anatomical band
    depth         when boxes overlap, prefer the one whose bottom edge is
                  lower, i.e. nearer the camera and therefore in front

A COST CEILING, DELIBERATELY
────────────────────────────
Assignments above `max_cost` are rejected and the face is left unassigned. This
is asymmetric on purpose: an unattached face costs coverage — the person stays
"Detecting..." for that frame — whereas a wrongly attached face mints or
reinforces an identity for the wrong human, and nothing downstream will catch
it. Refusing to guess is the safe failure.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Box = Sequence[float]          # [x, y, w, h]

# Anatomical band for face-height / person-height. A standing adult is roughly
# 7-8 head-heights; a detector face box covers less than the whole head, and a
# seated or partly-occluded person inflates the ratio, so the band is generous.
SCALE_LO, SCALE_HI = 0.06, 0.35
# Face centre as a fraction down the person box. Heads sit at the top; the band
# is wide because person boxes are often cropped at the bottom.
VERT_LO, VERT_HI = 0.0, 0.40

W_CONTAIN, W_VERT, W_SCALE, W_DEPTH = 1.0, 0.35, 0.35, 0.20


def _containment(face: Box, person: Box) -> float:
    """Fraction of the FACE box lying inside the person box."""
    fx, fy, fw, fh = face
    px, py, pw, ph = person
    ix = max(0.0, min(fx + fw, px + pw) - max(fx, px))
    iy = max(0.0, min(fy + fh, py + ph) - max(fy, py))
    area = fw * fh
    return (ix * iy) / area if area > 0 else 0.0


def _band_penalty(v: float, lo: float, hi: float) -> float:
    """0 inside [lo, hi], rising linearly outside, clipped at 1."""
    if lo <= v <= hi:
        return 0.0
    span = max(hi - lo, 1e-6)
    d = (lo - v) if v < lo else (v - hi)
    return float(min(1.0, d / span))


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy
    u = aw * ah + bw * bh - i
    return i / u if u > 0 else 0.0


def pair_cost(face: Box, person: Box, person_boxes: Optional[List[Box]] = None,
              frame_h: Optional[float] = None) -> float:
    """
    Cost of assigning `face` to `person`. Lower is better; 0 is a perfect fit.

    Containment dominates: a face mostly outside a box is almost certainly not
    that person's, whatever the other terms say.
    """
    fx, fy, fw, fh = face
    px, py, pw, ph = person

    contain = _containment(face, person)
    c_contain = 1.0 - contain

    face_cy = fy + fh / 2.0
    rel_v = (face_cy - py) / ph if ph > 0 else 1.0
    c_vert = _band_penalty(rel_v, VERT_LO, VERT_HI)

    ratio = fh / ph if ph > 0 else 1.0
    c_scale = _band_penalty(ratio, SCALE_LO, SCALE_HI)

    # Depth proxy: only meaningful where this box overlaps another. A box whose
    # bottom edge is lower stands nearer the camera and occludes the one behind,
    # so a face in the shared region more likely belongs to it.
    c_depth = 0.0
    if person_boxes:
        rivals = [b for b in person_boxes
                  if list(b) != list(person) and _iou(b, person) > 0.0]
        if rivals:
            lowest = max([py + ph] + [b[1] + b[3] for b in rivals])
            span = float(frame_h) if frame_h else max(1.0, lowest)
            c_depth = float(min(1.0, max(0.0, (lowest - (py + ph)) / max(span, 1e-6))))

    return (W_CONTAIN * c_contain + W_VERT * c_vert
            + W_SCALE * c_scale + W_DEPTH * c_depth)


def build_cost_matrix(faces: List[Box], person_boxes: List[Box],
                      frame_h: Optional[float] = None) -> np.ndarray:
    """(n_faces x n_person_boxes) cost matrix."""
    n, m = len(faces), len(person_boxes)
    C = np.zeros((n, m), dtype=np.float64)
    for i, f in enumerate(faces):
        for j, p in enumerate(person_boxes):
            C[i, j] = pair_cost(f, p, person_boxes, frame_h)
    return C


def assign(faces: List[Box], person_boxes: List[Box],
           max_cost: float = 0.60, frame_h: Optional[float] = None
           ) -> Dict[int, int]:
    """
    Optimal one-to-one face -> person assignment.

    Returns {person_index: face_index} — the direction callers want, since
    they iterate person boxes and ask which face belongs to each. It is also
    the only direction that can express the shipped rule's defect, where two
    boxes claim one face; a face-keyed dict would silently drop one of them.

    Person boxes whose best face exceeds `max_cost` are omitted, leaving that
    person unidentified for the frame — a safe outcome, unlike a misassignment.

    Falls back to greedy-by-cost if scipy is unavailable, so the feature
    degrades rather than failing the pipeline.
    """
    if not faces or not person_boxes:
        return {}
    C = build_cost_matrix(faces, person_boxes, frame_h)

    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(C)
        pairs = zip(rows.tolist(), cols.tolist())
    except Exception as exc:                        # noqa: BLE001
        logger.debug("face_assign: scipy unavailable (%s) — greedy fallback", exc)
        pairs = _greedy(C)

    return {int(j): int(i) for i, j in pairs if C[i, j] <= max_cost}


def _greedy(C: np.ndarray):
    """One-to-one greedy over ascending cost. Used only without scipy."""
    n, m = C.shape
    order = sorted(((C[i, j], i, j) for i in range(n) for j in range(m)))
    used_f, used_p = set(), set()
    for _c, i, j in order:
        if i in used_f or j in used_p:
            continue
        used_f.add(i); used_p.add(j)
        yield i, j


def match_faces_to_boxes(face_boxes: List[Box], person_boxes: List[Box],
                         cfg, frame_h: Optional[float] = None
                         ) -> Dict[int, int]:
    """
    Config-gated entry point.

    With `enable_optimal_face_assignment` off — the default — this is the
    shipped greedy rule: for each person box in order, the first UNCLAIMED face
    whose centroid lies in the upper 60% of that box. A face is consumed once
    (see the double-claim fix below); ordering and match criteria are otherwise
    unchanged.
    """
    if getattr(cfg, "enable_optimal_face_assignment", False):
        return assign(face_boxes, person_boxes,
                      max_cost=float(getattr(cfg, "face_assign_max_cost", 0.60)),
                      frame_h=frame_h)

    # Greedy, first-match-wins per box — the shipped rule, WITH the
    # double-claim defect fixed.
    #
    # Originally no box checked whether another had already taken the face, so
    # one face could be claimed by SEVERAL boxes at once and identify two
    # people simultaneously. Nothing downstream catches that: the face is real
    # and gate-passing, so the face veto, the ID-switch guard and the evidence
    # gate all see legitimate evidence and none of them objects.
    #
    # This is deliberately the MINIMAL fix. Nothing is reordered or re-ranked;
    # a box that would have stopped on an already-consumed face now continues
    # its scan instead. Optimal one-to-one assignment (the flag above) remains
    # a separate, larger change.
    out: Dict[int, int] = {}
    taken = set()
    for j, p in enumerate(person_boxes):
        px, py, pw, ph = p
        for i, f in enumerate(face_boxes):
            if i in taken:
                continue
            fcx, fcy = f[0] + f[2] / 2.0, f[1] + f[3] / 2.0
            if px <= fcx <= px + pw and py <= fcy <= py + ph * 0.6:
                out[j] = i
                taken.add(i)
                break
    return out
