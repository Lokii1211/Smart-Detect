# Identity evaluation metrics — exact definitions

Executable definitions live in [`eval/scoring.py`](scoring.py); this document
and that module are kept in lockstep. Formulas here are intended for verbatim
reuse in write-ups.

> **Ground truth only.** Every metric compares the code the system assigned
> against the *dataset's* label for that frame. Nothing reads `snapshots/`,
> the `persons` table, or any other system-produced artefact.
>
> This replaces the earlier snapshot-similarity check
> (`scripts/purity_eval.py`), which was **not a valid accuracy measure**:
> snapshots are only written when the evidence gate passes at
> ≥ `evidence_face_sim_threshold` face similarity, so its sample was filtered
> by the very property it measured. That script is now a regression check only.

---

## Notation

| Symbol | Meaning |
|---|---|
| `R` | all **assignment records**. Exactly one per (frame, ground-truth person) pair, **including frames where the detector produced no box** |
| `r.gt` | ground-truth identity for record `r` (the reference) |
| `r.code` | code the system assigned: `SDT-XXXX`, or the sentinel `"Detecting..."` |
| `r.tid` | ByteTrack tracker id, or `None` |
| `r.cam` | camera id |
| `r.order` | global monotonically increasing temporal index |
| `r.evidence_written` | a sighting row was actually written to the DB for this record |
| `C` | codes actually used: `C = { r.code : r ∈ R, r.code ≠ "Detecting..." }` |
| `gt(c)` | ground-truth identities code `c` touched |
| `maj(c)` | the identity code `c` was assigned to most often — what the code *empirically means* |

**One record per ground-truth person-frame is mandatory.** If a person is
present in the ground truth but the detector misses them, the record still
exists with `code = "Detecting..."` and `method = "no_detection"`. Omitting
those inflates coverage — in this project's own harness that bug turned a true
80.0% coverage into a reported 92.3%.

### Attribution of detections to ground truth

- Ground truth **with** boxes: greedy highest-IoU matching, one detection per
  ground-truth person, requiring **IoU ≥ 0.3**.
- Ground truth **without** boxes and exactly one person per frame
  (person-centric crops): the largest unused detection is that person's.
- Any ground-truth person left unmatched → `no_detection` record.

---

## 0. The bucket partition — foundation of everything below

Every record falls into **exactly one** of three buckets:

```
bucket(r) =
    UNASSIGNED    if r.code = "Detecting..."        (includes detector misses)
    CORRECT       if maj(r.code) = r.gt
    CONTAMINATED  otherwise
```

```
n_total        = |R|
n_correct      = |{ r : bucket(r) = CORRECT }|
n_contaminated = |{ r : bucket(r) = CONTAMINATED }|
n_unassigned   = |{ r : bucket(r) = UNASSIGNED }|
```

**Invariant, asserted at runtime in both `scoring.compute_all()` and
`run_sweep.py`:**

```
n_correct + n_contaminated + n_unassigned = n_total
frac_correct + frac_contaminated + frac_unassigned = 1.0
```

A run whose partition does not sum exactly to the total aborts rather than
emitting a table.

`UNASSIGNED` is additionally split for diagnosis — a detector miss and a
deliberate quality-gate refusal are different engineering problems:

```
n_unassigned_no_detection = |{ r : bucket(r)=UNASSIGNED, r.method = "no_detection" }|
n_unassigned_detecting    = n_unassigned − n_unassigned_no_detection
```

---

## 1. Identity Precision ↑

Of the person-frames the system *chose* to identify, how many are right.

```
identity_precision = n_correct / (n_correct + n_contaminated)
```

Range [0, 1]. Undefined (`n/a`) when nothing was assigned. Says nothing about
how much was skipped.

## 2. Coverage

Of all ground-truth person-frames, how many received any code.

```
coverage = (n_correct + n_contaminated) / n_total
```

Range [0, 1]. **Not "higher is better" in isolation.** The system withholds
identity below its face-quality gate (`face_quality_min_height_px`,
`face_quality_min_det_score`), so low coverage on distant or averted-face
footage is designed behaviour, not failure.

> ### ⚠️ Never combine metrics 1 and 2
> Do not report an F-score or any single blended number. The entire argument
> is the **trade-off** between precision and coverage: each mechanism buys
> precision by declining to answer. A blended score hides exactly the axis
> under study, and lets a system that guesses more look equal to one that
> guesses better.

## 3. Evidence Precision ↑ *(first-class)*

Of the sighting rows the system actually **wrote to the database**, what
fraction sit under the correct code.

```
E = { r ∈ R : r.evidence_written }
evidence_precision = |{ r ∈ E : bucket(r) = CORRECT }| / |E|
evidence_yield     = |E| / n_total
```

Range [0, 1]; undefined when no evidence was written.

This is what an operator or investigator actually sees — the persisted
evidence trail, not the transient on-screen label. It is the metric
`enable_evidence_gating` exists to improve, and the **only** metric in this
set that can distinguish ablation configs C and D, because every other metric
scores code *assignment* while the gate governs whether evidence is *written*.

Report `evidence_yield` beside it, for the same reason coverage accompanies
precision: a gate that writes almost nothing trivially scores 100%.

**Write-rate note.** Production additionally suppresses repeat sightings of a
code within 30 s of wall-clock time. Offline this is controlled by
`--sighting-dedup-seconds`, **default 0** (one row per gate-passing frame),
because adapter timestamps are synthetic (`frame_index / fps`) so a 30 s
window would be arbitrary and would collapse the sample. Dedup is a write-rate
limiter, not an accuracy mechanism: it does not systematically change the
correct:incorrect ratio, only the sample size. The setting used is recorded in
`metrics.json` as `sighting_dedup_seconds`.

## 4. Identity Purity ↑

Code-level: fraction of minted codes that refer to exactly one real person.

```
identity_purity = |{ c ∈ C : |gt(c)| = 1 }| / |C|
```

Range [0, 1]; undefined when `|C| = 0`. Counts **how many** identities are
polluted, not how badly — one stray frame condemns a whole code here but
barely moves contamination. Report with the contamination fraction.

## 5. Fragmentation ↓

Mean distinct codes per ground-truth person.

This is **not** the duplicate rate — see §6, which separates genuine
over-splitting from stray contaminated frames. Fragmentation counts both.

```
fragmentation = ( Σ_{p ∈ P} |{ r.code : r ∈ R, r.gt = p, r.code ≠ "Detecting..." }| ) / |P|
P = { r.gt : r ∈ R, r.code ≠ "Detecting..." }
```

**Ideal exactly 1.0.** Undefined when `|P| = 0`.

> **Never read alone.** A system minting a fresh code per frame scores perfect
> precision-by-construction and terrible fragmentation. A system putting
> everyone under one code scores perfect fragmentation (1.0) and catastrophic
> precision. This project's own ablation demonstrates the second failure: the
> pre-hardening config scores a *perfect* 1.000 fragmentation while being
> entirely wrong.

## 6. Duplicate Identities ↓

Duplicate identities per ground-truth person — the metric the registration
pose gate targets.

Distinct from `fragmentation` (§5), which counts **every** distinct code a
person was ever assigned and therefore also counts single stray frames caused
by contamination. A person with 80 frames under code `X` and 1 misassigned
frame under someone else's code `Y` has fragmentation 2.0 but **zero**
duplicates — `Y` is not their identity, it is an error already counted by
contamination.

A code counts as belonging to a person only when that person is the code's
**majority owner**:

```
owned(p)      = { c ∈ C : maj(c) = p }
duplicates(p) = max(0, |owned(p)| − 1)
duplicate_identities = ( Σ_{p ∈ owners} duplicates(p) ) / |owners|
owners = { p : |owned(p)| ≥ 1 }
```

**Ideal exactly 0.0.** Undefined when no code has a majority owner.

Note the denominator is `|owners|`, **not** the number of ground-truth people:
a person who was never successfully identified owns no code and does not
dilute the mean.

Reported alongside the raw counts (`total_duplicates`,
`people_with_duplicates`, `per_person`, `codes_owned`) so one badly-split
person cannot hide behind an average.

> **Read with §5, never instead of it.** Fragmentation and duplicates disagree
> by design. On this project's own worked example — one code spanning two
> people plus one person holding two codes — fragmentation is 1.500 while
> duplicates is 1.000. Quoting either alone misdescribes the system.

## 7. ID Switches ↓

Times a track's assigned code changes to a *different* code.

```
For each tracker id t:
    S_t = [ r ∈ R : r.tid = t, r.code ≠ "Detecting..." ] sorted by r.order
    switches(t) = |{ i : S_t[i].code ≠ S_t[i−1].code }|
id_switches = Σ_t switches(t)
```

Transitions to/from `"Detecting..."` are **excluded** (coverage, not a switch).
Records with `tid = None` are excluded. Not purely "lower is better": the
ID-switch guard deliberately causes a switch when it corrects a hijacked
track, so a more correct config can show *more* switches.

## 8. Cross-camera Re-association Accuracy ↑

Of people seen on more than one camera, how often the later camera reuses the
earlier camera's code.

```
For person p with cameras C_p (|C_p| ≥ 2), ordered by first appearance:
    dom(p, c) = most frequent code assigned to p on camera c
    for each consecutive ordered pair (c_i, c_{i+1}):
        pair += 1;  hit += 1 if dom(p, c_i) = dom(p, c_{i+1})
cross_camera_accuracy = hit / pair
```

Range [0, 1]. Undefined (`n/a`) when no person appears on two cameras —
reported as `n/a`, **never 0**: a single-camera dataset cannot fail this.

## 9. Runtime

Wall-clock around the full per-frame pipeline (detection → tracking → face
scan → arbitration → evidence gate), single-threaded and synchronous:

```
mean_frame_latency_ms   = 1000 · (Σ_f t_f) / N
median_frame_latency_ms = 1000 · median(t_f)
throughput_fps          = N / Σ_f t_f
```

This is **analysis** throughput, not stream fps. Hardware is recorded in
`metrics.json → runtime.machine`.

---

## Scope caveats that belong in any write-up

1. **The head-zoom second pass is not exercised offline** — it requires
   capture state from `LiveStream.start()`. It is a face-*recall* booster, so
   offline coverage is a **lower bound** relative to production; precision is
   unaffected in kind.
2. **Precision, coverage and fragmentation must be reported together.** No two
   of them alone distinguish a correct system from a degenerate one.
3. **Metrics are undefined, not zero, on empty inputs** — `n/a` propagates to
   the summary table rather than a misleading `0.000`.
4. **`maj(c)` is empirical**, computed from the run being scored. A code
   assigned to two people equally splits by first-encountered tie-break; with
   small samples this makes purity and contamination coarse.
