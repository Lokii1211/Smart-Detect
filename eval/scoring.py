"""
eval/scoring.py
────────────────
Metric computation for the offline identity evaluation.

GROUND TRUTH ONLY. Nothing in this module reads snapshots/, the persons
table, or any other system-produced artefact. Every metric compares the
code the system assigned against the dataset's label for that frame.

This replaces scripts/purity_eval.py as the accuracy measure. purity_eval
scored the face-similarity of saved snapshots, but snapshots are only
written when the evidence gate passes at >= evidence_face_sim_threshold —
so its sample was filtered by the very property it measured. It is retained
only as a regression check (see its header).

Input is a list of Assignment records from eval/run_eval.py: exactly one
record per (frame, ground-truth person) pair, including frames where the
detector missed the person entirely. Formulas are documented verbatim in
eval/METRICS.md; this module is the executable definition of that document.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

# SmartIdentifier's sentinel for "no identity assigned".
UNASSIGNED = "Detecting..."

# z for a two-sided 95% interval on the standard normal.
Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> Dict:
    """
    Wilson score interval for a binomial proportion.

        centre = (p̂ + z²/2n) / (1 + z²/n)
        half   = ( z / (1 + z²/n) ) · sqrt( p̂(1−p̂)/n + z²/4n² )
        CI     = centre ± half            (clipped to [0, 1])

    Wilson rather than the normal-approximation (Wald) interval because Wald
    is degenerate exactly where this project's metrics live: at p̂ = 1.0 (a
    perfect precision) Wald gives a zero-width interval, which is nonsense at
    n = 30. Wilson stays inside [0,1] and remains sensible at the extremes and
    at small n.

    Returns point estimate, bounds, half-width, and n. `n = 0` -> all None:
    a proportion with no denominator is undefined, not zero.
    """
    if n <= 0:
        return {"value": None, "lo": None, "hi": None, "half_width": None, "n": 0}
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    # The Wilson interval always contains p̂ analytically (at p̂=1, centre+half
    # reduces to exactly 1.0), but floating-point rounding can land a hair
    # below — e.g. 30/30 gives hi=0.9999999999999999. Clamp so the invariant
    # 0 <= lo <= p̂ <= hi <= 1 holds exactly.
    lo = min(p, max(0.0, centre - half))
    hi = max(p, min(1.0, centre + half))
    return {
        "value": p,
        "lo": lo,
        "hi": hi,
        "half_width": half,
        "n": n,
    }


def fmt_ci(ci: Dict, pct: bool = True, nd: int = 1) -> str:
    """'97.2% [85.8-99.5]' — the only form in which a proportion should be
    reported. A bare point estimate is not a result."""
    if ci is None or ci.get("value") is None:
        return "n/a"
    if pct:
        return (f"{100*ci['value']:.{nd}f}% "
                f"[{100*ci['lo']:.{nd}f}-{100*ci['hi']:.{nd}f}]")
    return f"{ci['value']:.3f} [{ci['lo']:.3f}-{ci['hi']:.3f}]"

# Bucket labels — every record gets exactly one.
CORRECT      = "CORRECT"
CONTAMINATED = "CONTAMINATED"
NO_ASSIGN    = "UNASSIGNED"


@dataclass
class Assignment:
    """
    One scored observation = one ground-truth person in one frame.

    A record exists for EVERY ground-truth person-frame, including those
    where the detector produced no box at all (code=UNASSIGNED,
    method="no_detection"). Without that, detection misses silently leave
    the denominator and coverage is overstated.
    """
    frame_id:    str
    camera_id:   str
    seq_id:      str
    gt_person:   str            # ground-truth identity (the reference)
    code:        str            # SDT-XXXX, or UNASSIGNED
    tracker_id:  Optional[int]
    method:      str            # face | dress_color | body_structure |
                                # new_registration | pending | no_detection
    order:       int            # global monotonic temporal index
    evidence_written: bool = False   # a sighting row was actually written
                                     # for this record (production log_sighting)


# ─── Bucket assignment: the foundation of every rate below ──────────────────

def _majority_gt_per_code(records: List[Assignment]) -> Dict[str, str]:
    """
    majority(c) = the ground-truth identity that code c was assigned to most
    often. This defines what a code "means" empirically, so a code can be
    judged right or wrong per frame. Ties resolve to the first-encountered
    identity (Counter.most_common is stable on insertion order).
    """
    per_code: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r.code != UNASSIGNED:
            per_code[r.code][r.gt_person] += 1
    return {c: cnt.most_common(1)[0][0] for c, cnt in per_code.items()}


def classify(records: List[Assignment]) -> Dict[str, str]:
    """
    frame-key -> bucket, for every record. Exhaustive and mutually exclusive:

        UNASSIGNED    if code == "Detecting..."   (incl. detection misses)
        CORRECT       if majority(code) == gt
        CONTAMINATED  otherwise
    """
    majority = _majority_gt_per_code(records)
    out: Dict[str, str] = {}
    for i, r in enumerate(records):
        key = f"{i}:{r.frame_id}:{r.gt_person}"
        if r.code == UNASSIGNED:
            out[key] = NO_ASSIGN
        elif majority.get(r.code) == r.gt_person:
            out[key] = CORRECT
        else:
            out[key] = CONTAMINATED
    return out


def buckets(records: List[Assignment]) -> Dict:
    """
    The three-way partition of all ground-truth person-frames.

        n_correct + n_contaminated + n_unassigned == n_total   (asserted)

    Rates are each over n_total, so the three fractions sum to exactly 1.
    """
    total = len(records)
    if total == 0:
        return {"n_total": 0, "n_correct": 0, "n_contaminated": 0,
                "n_unassigned": 0, "frac_correct": None,
                "frac_contaminated": None, "frac_unassigned": None,
                "n_unassigned_no_detection": 0, "n_unassigned_detecting": 0}

    cls = classify(records)
    counts = Counter(cls.values())
    n_correct      = counts[CORRECT]
    n_contaminated = counts[CONTAMINATED]
    n_unassigned   = counts[NO_ASSIGN]

    # Hard invariant — a silent leak here invalidates every rate below.
    assert n_correct + n_contaminated + n_unassigned == total, (
        f"bucket partition is not exhaustive: {n_correct} + {n_contaminated} "
        f"+ {n_unassigned} != {total}"
    )

    # Diagnostic split of the UNASSIGNED bucket: a detector miss and a
    # deliberate quality-gate refusal are different failures.
    no_det = sum(1 for r in records
                 if r.code == UNASSIGNED and r.method == "no_detection")
    return {
        "n_total":            total,
        "n_correct":          n_correct,
        "n_contaminated":     n_contaminated,
        "n_unassigned":       n_unassigned,
        "frac_correct":       n_correct / total,
        "frac_contaminated":  n_contaminated / total,
        "frac_unassigned":    n_unassigned / total,
        # Each bucket share is a binomial proportion over the same n_total.
        # NB: the three intervals are NOT independent and do not sum to 1 —
        # they are marginal intervals on a multinomial, quoted per bucket.
        "ci_correct":         wilson(n_correct, total),
        "ci_contaminated":    wilson(n_contaminated, total),
        "ci_unassigned":      wilson(n_unassigned, total),
        "n_unassigned_no_detection": no_det,
        "n_unassigned_detecting":    n_unassigned - no_det,
        "sums_to_one": True,
    }


# ─── The two headline numbers — REPORTED SEPARATELY, NEVER COMBINED ─────────

def identity_precision(records: List[Assignment]) -> Dict:
    """
    Of the person-frames the system chose to identify, how many are right.

        precision = n_correct / (n_correct + n_contaminated)

    Undefined (None) when the system assigned nothing. Says nothing about
    how much was skipped — that is coverage's job. Do NOT combine the two
    into an F-score: the design deliberately trades coverage for precision,
    and a single number hides exactly the axis under study.
    """
    b = buckets(records)
    assigned = b["n_correct"] + b["n_contaminated"]
    ci = wilson(b["n_correct"], assigned)
    return {
        "value": ci["value"],
        "ci": ci,
        "n_correct": b["n_correct"],
        "n_assigned": assigned,
    }


def identity_coverage(records: List[Assignment]) -> Dict:
    """
    Of all ground-truth person-frames, how many received any code.

        coverage = (n_correct + n_contaminated) / n_total

    NOT "higher is better" alone. SmartDetect withholds identity below its
    face-quality gate, so low coverage on distant/averted footage is designed
    behaviour. Coverage bought by guessing appears as reduced precision.
    """
    b = buckets(records)
    assigned = b["n_correct"] + b["n_contaminated"]
    ci = wilson(assigned, b["n_total"])
    return {
        "value": ci["value"],
        "ci": ci,
        "n_assigned": assigned,
        "n_total": b["n_total"],
        "n_unassigned_no_detection": b["n_unassigned_no_detection"],
        "n_unassigned_detecting": b["n_unassigned_detecting"],
    }


def evidence_precision(records: List[Assignment]) -> Dict:
    """
    FIRST-CLASS METRIC. Of the sighting rows the system actually wrote to
    the database, what fraction sit under the correct code.

        E = { r : r.evidence_written }
        evidence_precision = |{ r in E : bucket(r) == CORRECT }| / |E|

    This is what an operator or investigator actually sees — the stored
    evidence trail, not the transient on-screen label. It is the metric
    enable_evidence_gating exists to improve, and the only one of the set
    that can distinguish configs C and D.

    Reported alongside evidence_yield (how much evidence was written at
    all), for the same reason precision and coverage are kept separate.
    """
    cls = classify(records)
    written, correct = 0, 0
    for i, r in enumerate(records):
        if not r.evidence_written:
            continue
        written += 1
        if cls[f"{i}:{r.frame_id}:{r.gt_person}"] == CORRECT:
            correct += 1
    b = buckets(records)
    ci = wilson(correct, written)
    yield_ci = wilson(written, b["n_total"])
    return {
        "value": ci["value"],
        "ci": ci,
        "n_written": written,
        "n_correct": correct,
        "n_wrong": written - correct,
        "evidence_yield": yield_ci["value"],
        "evidence_yield_ci": yield_ci,
    }


# ─── Supporting structural metrics ──────────────────────────────────────────

def identity_purity(records: List[Assignment]) -> Dict:
    """
    Code-level: fraction of minted codes that refer to exactly one GT person.

        purity = |{ c : |gt(c)| == 1 }| / |C|

    Counts HOW MANY identities are polluted, not how badly. Pair with
    contamination (frame-level severity).
    """
    by_code: Dict[str, set] = defaultdict(set)
    for r in records:
        if r.code != UNASSIGNED:
            by_code[r.code].add(r.gt_person)
    if not by_code:
        return {"value": None, "n_codes": 0, "impure_codes": []}
    impure = sorted(c for c, ids in by_code.items() if len(ids) > 1)
    ci = wilson(len(by_code) - len(impure), len(by_code))
    return {
        "value": ci["value"],
        "ci": ci,
        "n_codes": len(by_code),
        "impure_codes": [{"code": c, "gt_ids": sorted(by_code[c])} for c in impure],
    }


def fragmentation(records: List[Assignment]) -> Dict:
    """
    Mean distinct codes per ground-truth person — the duplicate-identity cost.

        fragmentation = (1/|P|) * sum_{p in P} |{ codes assigned to p }|

    Ideal exactly 1.0. NEVER read alone: a system that mints one code per
    frame scores perfect precision-by-construction and terrible
    fragmentation; a system that puts everyone under one code scores perfect
    fragmentation and catastrophic precision.
    """
    by_person: Dict[str, set] = defaultdict(set)
    for r in records:
        if r.code != UNASSIGNED:
            by_person[r.gt_person].add(r.code)
    if not by_person:
        return {"value": None, "n_people": 0, "per_person": {}}
    counts = {p: len(cs) for p, cs in by_person.items()}
    return {"value": sum(counts.values()) / len(counts),
            "n_people": len(counts), "per_person": dict(sorted(counts.items()))}


def duplicate_identities(records: List[Assignment]) -> Dict:
    """
    Duplicate identities per ground-truth person — the metric the pose gate
    targets.

    Distinct from `fragmentation`, which counts EVERY distinct code a person
    was ever assigned and therefore also counts single stray frames caused by
    contamination. A person with 80 frames under code X and 1 misassigned
    frame under someone else's code Y has fragmentation 2.0 but ZERO
    duplicates — Y is not their identity, it is an error already counted by
    contamination_rate.

    A code counts as belonging to a person only when that person is the
    code's MAJORITY owner:

        owned(p) = { c : maj(c) == p }
        duplicates(p) = max(0, |owned(p)| - 1)
        duplicates_per_person = mean over people with >= 1 owned code

    Ideal 0.0. Reported alongside the raw counts so a single bad person
    cannot hide behind an average.
    """
    majority = _majority_gt_per_code(records)
    owned: Dict[str, set] = defaultdict(set)
    for code, person in majority.items():
        owned[person].add(code)
    if not owned:
        return {"value": None, "n_people": 0, "total_duplicates": 0,
                "per_person": {}, "people_with_duplicates": 0}
    per_person = {p: max(0, len(cs) - 1) for p, cs in owned.items()}
    total = sum(per_person.values())
    return {
        "value": total / len(per_person),
        "n_people": len(per_person),
        "total_duplicates": total,
        "people_with_duplicates": sum(1 for v in per_person.values() if v > 0),
        "per_person": dict(sorted(per_person.items())),
        "codes_owned": {p: sorted(cs) for p, cs in sorted(owned.items())},
    }


def id_switches(records: List[Assignment]) -> Dict:
    """
    Times a track's assigned code changes to a different code.

    Transitions to/from UNASSIGNED are excluded (that is coverage, not a
    switch). Untracked records excluded. Not purely "lower is better": the
    ID-switch guard deliberately causes one when correcting a hijacked track.
    """
    per_track: Dict[int, List[Assignment]] = defaultdict(list)
    for r in records:
        if r.code != UNASSIGNED and r.tracker_id is not None:
            per_track[r.tracker_id].append(r)
    total, detail = 0, {}
    for tid, rs in per_track.items():
        rs.sort(key=lambda r: r.order)
        n = sum(1 for a, b in zip(rs, rs[1:]) if a.code != b.code)
        if n:
            detail[str(tid)] = n
        total += n
    return {"value": total, "n_tracks": len(per_track), "per_track": detail}


def cross_camera_reassociation(records: List[Assignment]) -> Dict:
    """
    Of GT people seen on >1 camera, fraction whose code on the later camera
    equals their dominant code on the earlier one.

    Undefined (None) when no person appears on two cameras — reported as
    n/a, never as 0, since a single-camera dataset cannot fail this.
    """
    by_person_cam: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    first_seen: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in records:
        if r.code == UNASSIGNED:
            continue
        by_person_cam[r.gt_person][r.camera_id][r.code] += 1
        prev = first_seen[r.gt_person].get(r.camera_id)
        if prev is None or r.order < prev:
            first_seen[r.gt_person][r.camera_id] = r.order

    pairs = hits = 0
    detail = []
    for person, cams in by_person_cam.items():
        if len(cams) < 2:
            continue
        ordered = sorted(cams.keys(), key=lambda c: first_seen[person][c])
        dom = {c: cams[c].most_common(1)[0][0] for c in ordered}
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            pairs += 1
            ok = dom[a] == dom[b]
            hits += int(ok)
            detail.append({"person": person, "from_cam": a, "to_cam": b,
                           "from_code": dom[a], "to_code": dom[b], "match": ok})
    if pairs == 0:
        return {"value": None, "n_pairs": 0, "detail": [],
                "note": "no ground-truth person appeared on more than one camera"}
    ci = wilson(hits, pairs)
    return {"value": ci["value"], "ci": ci, "n_pairs": pairs, "detail": detail}


def compute_all(records: List[Assignment], runtime: Dict) -> Dict:
    b = buckets(records)
    # Re-assert at the top level so a malformed record set fails the run
    # rather than producing a plausible-looking table.
    if b["n_total"]:
        assert (b["n_correct"] + b["n_contaminated"] + b["n_unassigned"]
                == b["n_total"]), "bucket partition broken"
        s = b["frac_correct"] + b["frac_contaminated"] + b["frac_unassigned"]
        assert abs(s - 1.0) < 1e-9, f"bucket fractions sum to {s}, not 1.0"
    return {
        "buckets":               b,
        "identity_precision":    identity_precision(records),
        "identity_coverage":     identity_coverage(records),
        "evidence_precision":    evidence_precision(records),
        "identity_purity":       identity_purity(records),
        "fragmentation":         fragmentation(records),
        "duplicate_identities":  duplicate_identities(records),
        "id_switches":           id_switches(records),
        "cross_camera_reassoc":  cross_camera_reassociation(records),
        "runtime":               runtime,
        "n_records":             len(records),
    }
