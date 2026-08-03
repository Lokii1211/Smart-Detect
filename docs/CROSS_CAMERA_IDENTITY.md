# Cross-Camera Identity Resolution — Design

**Status:** DESIGN ONLY — nothing here is implemented.
**Date:** 2026-08-02
**Depends on:** `docs/SCALE_ARCHITECTURE.md` (worker pool, PostgreSQL), `eval/METRICS.md`
**Prerequisite:** a labelled multi-camera dataset. See §7.0 — this design cannot be validated without one.

---

## 0. Summary

Cross-camera identity today is **accidental, not designed**. There is no service that owns it; it emerges from every camera querying one shared face gallery. That produces four concrete defects (§1), all of which this design addresses by separating two things that are currently conflated:

- **Track continuity** — per-camera, per-frame, owned by ByteTrack. Stays where it is.
- **Identity** — global, per-tracklet, owned by a new service. Moves out of `LiveStream`.

The single most valuable change is **§4, tracklet-level voting**: it is simultaneously cheaper (one gallery search per tracklet instead of per cache-miss), more accurate (aggregate K best frames instead of trusting the first adequate one), and it removes the root cause the ID-switch guard exists to patch.

---

## 1. What is wrong today — precisely

Verified in code, not assumed:

```
grep "active_streams" backend/main.py | grep -i identif   ->  no matches
```

Cameras never consult each other. Each `LiveStream` calls `SmartIdentifier.identify()`, which calls `find_person_by_embedding()` against the whole `persons` table. Cross-camera association is a side effect of a shared gallery.

### Defect 1 — concurrent double-assignment
`cameras/live_stream.py:620` builds `claimed_codes` from **this camera's** tracks only:

```python
current_tids  = {t for t, _ in tracked if t is not None}
claimed_codes = {self._track_codes[t]["code"] for t in current_tids if t in self._track_codes}
```

Camera A and camera B can hand `SDT-0042` to two different people in the same instant. Nothing detects it. The `exclude_codes` guarantee is per-frame, per-camera.

### Defect 2 — no spatiotemporal feasibility
A person last seen at the north gate 2 seconds ago can be matched at a camera 400 m away. The only temporal constraint is `reid_reassoc_window_hours` (12 h) and `colour_reassoc_window_minutes` (10 min) — both are *recency* windows, not *reachability* constraints. Physically impossible transitions are accepted whenever appearance happens to match.

### Defect 3 — identity decided on the first adequate frame
`identify()` runs on tracker-cache-miss and the result is cached for the life of the track. Whichever frame first yields a gate-passing face decides the identity — the noisiest possible estimator. The ID-switch guard (`_apply_id_switch_guard`) exists specifically to detect when that decision was wrong, which is treating the symptom.

### Defect 4 — no global arbitration
There is no component that can answer "is this the same person as that?" It is implicit in a `SELECT ... ORDER BY distance LIMIT 1`, so there is no place to add topology, priors, or conflict resolution.

**Measured consequence:** fragmentation 1.5 codes/person, and cross-camera accuracy currently rests on **n=1 transition** — statistically meaningless (`eval/results/perf/ablation_*`).

---

## 2. Item 1 — Global Identity Service (GIS)

### Boundary

```
┌─────────────────────────── per camera (unchanged) ────────────────────────┐
│ capture → YOLO → ByteTrack → face scan → quality gate                     │
│                                                                            │
│ owns: tracker_id continuity, _track_age, face-bbox smoothing               │
│ owns NO SDT codes                                                          │
└──────────────────────────────────┬─────────────────────────────────────────┘
                                   │  TrackletObservation (on close or K-frames)
                                   ▼
┌──────────────────────── Global Identity Service ──────────────────────────┐
│  candidate generation (gallery ANN)                                        │
│      ↓                                                                     │
│  spatiotemporal feasibility filter   (§3 topology + priors)                │
│      ↓                                                                     │
│  appearance scoring  (face-anchored; colour/re-ID subordinate)             │
│      ↓                                                                     │
│  global arbitration  (no code to two live tracklets; conflict resolution)  │
│      ↓                                                                     │
│  decision: MATCH <code> | NEW <code> | DEFER                               │
│                                                                            │
│  owns: persons gallery, code allocation, cross-camera state                │
└────────────────────────────────────────────────────────────────────────────┘
```

### The contract

```python
@dataclass(frozen=True)
class TrackletObservation:
    tracklet_id:   str            # globally unique: f"{camera_id}:{tracker_id}:{epoch}"
    camera_id:     str
    zone_id:       str
    t_start:       float          # first frame, epoch seconds
    t_end:         float          # last frame (or 'still open')
    n_frames:      int
    face_evidence: list[FaceSample]   # K best, quality-ranked (§4)
    reid_vector:   np.ndarray | None  # one per tracklet, not per frame
    dress_hsv:     dict | None
    is_closed:     bool

@dataclass(frozen=True)
class FaceSample:
    embedding:  np.ndarray
    det_score:  float
    height_px:  float
    yaw_pitch:  tuple[float, float] | None   # from InsightFace 5-pt landmarks
    t:          float

@dataclass(frozen=True)
class IdentityDecision:
    tracklet_id: str
    code:        str | None       # None => DEFER
    status:      str              # matched | registered | deferred | rejected
    confidence:  float
    method:      str              # face | face+topology | reid | new
    rationale:   dict             # candidates considered, why each was rejected
```

`rationale` is not decoration. Under `docs/DATA_GOVERNANCE.md` an operator must be able to answer *why* a person was identified; today that answer does not exist.

### Deployment

**Phase A — in-process singleton.** One GIS object, `asyncio` or a lock-protected queue, called by all workers in the same process. No network. This is where it should start.

**Phase B — separate service** only when workers are on multiple hosts (`SCALE_ARCHITECTURE.md` Phase 4). gRPC or HTTP; single writer for code allocation.

> **Do not start at Phase B.** A network hop per tracklet is fine (tracklets are seconds apart, unlike frames), but a distributed service is a distributed-consensus problem for code allocation. Earn it.

### Concurrency

Code allocation and "is this code already live?" must be serialised. Options:

| Mechanism | Verdict |
|---|---|
| Single-writer GIS (Phase A) | ✅ Simplest correct answer |
| PostgreSQL advisory lock per candidate code | ✅ Works multi-host; needs the pgvector fix first |
| Optimistic + reconcile | ❌ Reconciling identity after evidence is written is a governance problem |

---

## 3. Items 2 & 3 — Topology graph and transition priors

### Model

Directed graph. Nodes are **zones** (not cameras — two cameras may cover one zone, and a person moving between them is not a transition).

```python
@dataclass
class Transition:
    src: str; dst: str
    t_min:    float      # hard floor: fastest physically possible (walk/run)
    t_median: float
    t_p95:    float
    prob:     float      # P(dst | leaving src), sums <= 1 with an 'exit' sink
    bidirectional: bool
```

Stored in a `zone_transitions` table (operator-editable), not in code.

### How priors constrain matching

The filter runs **before** appearance scoring, to shrink the candidate set:

```
feasible(candidate P, observation O):
    last = P.last_seen           # (zone, t)
    if last is None:                       return True          # never seen: no constraint
    dt = O.t_start - last.t
    if dt < 0:                             return False         # out of order
    if last.zone == O.zone:                return True          # same zone: no travel needed
    edge = topology.path(last.zone, O.zone)
    if edge is None:                       return REJECT_UNREACHABLE
    if dt < edge.t_min:                    return REJECT_TELEPORT
    return True

prior(candidate, O) = log P(dt | edge)      # lognormal fitted to observed transitions
```

Final score:

```
score = w_face · face_similarity  +  w_topo · prior      (w_topo << w_face)
```

### Three rules that must not be violated

1. **Topology never mints an identity.** It only *removes* candidates. A strong prior with a weak face must never produce a match — that would invert the face-anchored principle that `tests/test_identity_arbitration.py` pins.
2. **Rejection is logged, not silent.** `REJECT_TELEPORT` is the most diagnostic signal the system can emit: it means appearance matching *would* have made an impossible claim. Feed it to the audit log and to §7's metrics.
3. **Unknown topology is permissive, not restrictive.** A missing edge on an unsurveyed site must degrade to today's behaviour, not lock the system out. `topology.path()` returns `None` only when the operator has explicitly declared zones non-adjacent.

### Cold start

Topology cannot be guessed. Three stages:

1. **Manual** — operator draws adjacency and sets `t_min` from a walk test. ~1 hour per site.
2. **Learned `t_min`/`t_median`** — from confirmed same-identity transitions (high face confidence only, else you learn your own errors). Needs ~50 transitions/edge before displacing the manual value.
3. **Learned structure** — infer edges from co-occurrence. Only with substantial data; low priority.

> **Feedback-loop hazard:** learning priors from the system's own matches will happily confirm its own mistakes. Learn only from transitions confirmed by high-confidence face matches (≥ 0.8), and hold out a labelled set to check the learned distribution against truth.

---

## 4. Item 4 — Tracklet-level identity voting

**The highest-value item.** Cheaper *and* more accurate, and it removes the cause the ID-switch guard patches.

### Today

```
frame N:   no gate-passing face      -> "Detecting..."
frame N+1: gate-passing face         -> identify() -> SDT-0042 -> CACHED FOREVER
frame N+2..: cache hit               -> no arbitration at all
```

Identity rests on **one frame**, chosen by "first to pass the gate" — not "best". `_apply_id_switch_guard` then spends two more contradicting frames detecting that the choice was wrong.

### Proposed

Maintain a bounded quality-ranked buffer per tracklet:

```
quality(face) = w1·norm(det_score) + w2·norm(height_px) + w3·frontality(yaw,pitch)
```

`frontality` comes from the 5-point landmarks InsightFace already returns — no extra inference. (This also addresses the duplicate-identity cause found earlier: the same woman registered twice because a head-down frame was used for registration.)

Decide once, from the top-K (K≈5):

```
consensus = normalise(Σ_{i∈topK} w_i · e_i)        # quality-weighted mean embedding
```

Search the gallery **once** with `consensus`. Averaging K independent noisy observations of the same face reduces embedding noise ~√K.

### Two-phase output — resolves the latency objection

Voting needs K frames, but the UI must label a box immediately. This maps exactly onto the existing evidence gate:

| Phase | When | Used for | Authority |
|---|---|---|---|
| **Provisional** | first gate-passing face | on-screen label only | none — may change |
| **Confirmed** | K frames, or tracklet close | sightings, snapshots, trails, alerts | authoritative |

This is *already the system's philosophy* — `_evidence_gate_ok()` withholds evidence until a face confirms the code. Tracklet voting generalises it: **display may be provisional; evidence must be confirmed.**

### Why cheaper

| | Today | Voting |
|---|---|---|
| Gallery searches | one per cache-miss (plus one per ID-switch re-identify) | **one per tracklet** |
| OSNet calls | one per registration/re-association | one per tracklet |
| Arbitration | per cache-miss | once |

Measured context: arbitration is 10.8 ms/frame at 0.10 calls/frame. On a 100-frame tracklet, today's path may run it several times (initial + each guard-triggered re-identify); voting runs it once.

### Interaction with the ID-switch guard

The guard becomes a **tracklet splitter** rather than a code corrector: a sustained face contradiction means ByteTrack merged two people, so close the tracklet and open a new one. Cleaner, and it stops the guard from having to reason about identity at all.

> **Ablation-critical:** `enable_id_switch_guard` semantics change. The existing config flag and its tests must be re-derived, and A/B/C/D re-run. Do not silently redefine it.

---

## 5. Interaction with existing guarantees

Every property currently pinned by tests must survive:

| Guarantee | Where pinned | Under this design |
|---|---|---|
| Face-anchored: unmatched face blocks colour/re-ID | `test_identity_arbitration.py` | **Preserved** — GIS applies it at tracklet level |
| Registration requires a face | 4 ablation configs | **Preserved** — consensus needs ≥1 gate-passing face |
| Evidence face-confirmed only | `_evidence_gate_ok` | **Strengthened** — evidence only on confirmed decisions |
| Lookalike band [0.45, 0.56) blocked | `test_lookalike_band_*` | **Preserved**; topology adds a second filter |
| Erasure completeness | `test_governance.py` | ⚠️ **New surface** — GIS caches must be invalidated on erase (§6) |
| Purity/contamination | `eval/scoring.py` | Must not regress — §7 |

---

## 6. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **GIS cache outlives an erased identity** | 🔴 Compliance | `DELETE /persons/{code}` must invalidate GIS state synchronously. Extend `test_governance.py` to assert the code cannot be matched after erasure. |
| Wrong topology silently blocks real matches | 🟠 High | Permissive default; log every `REJECT_UNREACHABLE`; alert on rate spikes. |
| Voting delays identity by K frames | 🟠 Med | Two-phase provisional/confirmed. Measure time-to-confirmed (§7). |
| Priors learn from own errors | 🟠 Med | Learn only from ≥0.8 face matches; validate against held-out labels. |
| GIS is a bottleneck/SPOF | 🟡 Med | Per-tracklet (seconds), not per-frame. Degrade to per-camera behaviour if unavailable. |
| Large refactor, weak eval data | 🔴 High | **§7.0 blocks the work.** |

---

## 7. Evaluation plan

### 7.0 Prerequisite — the dataset (blocking)

Current corpus: **2 identities, 45 frames, 1 cross-camera transition.** Cross-camera accuracy of "100%" rests on a single observation. **No cross-camera design can be evaluated on this.**

Required (extends `eval/validate_dataset.py` gates):

| Property | Minimum | Why |
|---|---|---|
| Identities | 15 | existing gate |
| Labelled frames | 1000 | existing gate |
| **Cross-camera transitions** | **≥100** (raise from 10) | Wilson half-width ~±5pp at 90%; 10 gives ±20pp |
| Cameras | ≥3 | 2 cameras cannot exercise topology |
| **Impossible-transition pairs** | ≥20 | Cannot measure teleport rejection without them |
| Timestamps | real, synchronised | Priors are meaningless with synthetic time |

**ChokePoint fits**: 3 cameras per portal, per-subject labels, real timestamps — the adapter is already written (`eval/adapters/chokepoint.py`, parser-tested). Its P1E/P1L (enter/leave) structure gives genuine cross-camera transitions.

⚠️ **The FolderAdapter's synthetic timestamps (`frame_index / fps`) cannot evaluate transition priors.** Real capture time is required.

### 7.1 New metrics

Added to `eval/scoring.py`, all with Wilson intervals:

| Metric | Definition | Direction |
|---|---|---|
| `cross_camera_precision` | of cross-camera links asserted, fraction joining the same GT person | ↑ |
| `cross_camera_recall` | of true transitions, fraction the system linked | ↑ |
| `teleport_rejections` | impossible transitions correctly refused / total impossible | ↑ |
| `false_topology_blocks` | valid transitions wrongly refused | ↓ **critical** |
| `time_to_confirmed_ms` | tracklet start → confirmed identity | ↓ |
| `provisional_flip_rate` | fraction of provisional labels the confirmed decision overturns | ↓ |
| `gallery_searches_per_tracklet` | cost proxy | ↓ |

Precision **and** recall separately — never an F-score. Same reasoning as `METRICS.md` §2: topology buys precision by refusing to answer, and one blended number hides exactly that trade.

### 7.2 Ablation grid

Extends `config/ablation/`, each independently switchable and independently verified (`scripts/verify_flags.py`):

| Config | voting | topology | priors | Question |
|---|---|---|---|---|
| **E_current** | ✗ | ✗ | ✗ | today's behaviour — control |
| **F_voting** | ✓ | ✗ | ✗ | Does voting alone improve accuracy and cost? |
| **G_topology** | ✓ | ✓ | ✗ | Does hard reachability help beyond voting? |
| **H_priors** | ✓ | ✓ | ✓ | Do soft time priors add anything over hard gating? |
| **I_priors_only** | ✗ | ✓ | ✓ | Is topology useful without voting? (isolates interaction) |

Success = **F beats E on precision at no coverage cost**, and G/H reduce `teleport_rejections` misses **without** raising `false_topology_blocks`.

### 7.3 Regression gates (must hold at every step)

Run the existing suite unchanged:

- `identity_precision`, `contamination`, `identity_purity` — **must not regress** (`eval/run_sweep.py`)
- `fragmentation` — expected to *improve* (cross-camera linking should reduce duplicate codes); a rise means the design failed its main promise
- 100-test suite green; `scripts/run_tests.sh` coverage gate holds
- `python scripts/audit_routes.py --check` passes

### 7.4 Targeted stress tests

Synthetic scenarios with known-correct answers, as unit tests:

1. **Teleport** — same face at zones 400 m apart, 1 s apart. Expect: rejected, `REJECT_TELEPORT` logged.
2. **Legitimate handover** — A→B at median transition time. Expect: linked, same code.
3. **Simultaneous double-claim** — two cameras, same code candidate, same instant. Expect: exactly one wins; the other DEFERs or registers new.
4. **Twin problem** — two different people, near-identical appearance, incompatible topology. Expect: topology separates them. *This is the case only this design can solve.*
5. **Erasure invalidation** — erase mid-tracklet. Expect: code never re-emitted; GIS cache purged.

### 7.5 Latency budget

Voting must not regress the 69.1 ms/frame measured after the CoreML work:

| Path | Budget |
|---|---|
| Per-frame (no identity work) | ≤ 60 ms |
| Per-tracklet GIS decision | ≤ 50 ms (amortised ≪1 ms/frame) |
| Topology feasibility check | ≤ 1 ms (in-memory graph) |
| Time-to-confirmed | ≤ 2 s at 12 fps (K=5 + quality wait) |

---

## 8. Effort

| Item | Days | Risk |
|---|---|---|
| **7.0 Multi-camera labelled dataset** (ChokePoint or recorded) | 3 | med — **blocks all below** |
| Tracklet aggregation + quality ranking + `TrackletObservation` | 4 | med |
| GIS Phase A (in-process): candidates → score → arbitrate → decide | 6 | high |
| Extract identity out of `LiveStream`; two-phase provisional/confirmed | 5 | **high** — most stateful code |
| Topology model, storage, editor API, `t_min` walk-test tooling | 3 | low |
| Transition priors + learning pipeline + feedback-loop safeguards | 4 | med |
| New metrics + 5 ablation configs + flag verification | 3 | low |
| Stress tests (§7.4) + governance invalidation tests | 2 | low |
| **Total** | **30 d** | |

### Sequencing — each step independently shippable

1. **Dataset** (3 d) — blocks everything.
2. **Voting only** (9 d) — F vs E. Biggest single win; no topology needed. **Ship or abandon on this result.**
3. **Topology hard gating** (4 d) — G. Cheap once voting exists.
4. **Learned priors** (5 d) — H. Only if G shows headroom.
5. **GIS extraction / multi-host** (9 d) — only with `SCALE_ARCHITECTURE.md` Phase 3.

> If step 2 does not beat the control on the real dataset, **stop**. Topology and priors are refinements on a voting foundation; without it they add complexity to a per-frame estimator that is the actual problem.

---

## 9. Explicit non-goals

- **No per-frame cross-camera queries.** Tracklet granularity only; per-frame would put a global lookup in a 69 ms budget.
- **No appearance-only cross-camera matching at scale.** With 100k identities, appearance alone will produce lookalike collisions; topology is what makes it tractable.
- **No topology-only identity.** Location is never sufficient evidence.
- **No re-identification across days from clothing.** `reid_reassoc_window_hours` (12 h) stays; OSNet tracks clothing, not people.

---

## 10. Open questions

1. **Zone vs camera as the topology node?** Zones are more correct but need a survey. Cameras are available immediately. *Leaning: zones, defaulting one-zone-per-camera.*
2. **K for voting?** K=5 is a guess. Sweep on the real dataset.
3. **Does the ID-switch guard survive as a flag, or fold into tracklet splitting?** Affects existing configs and tests — decide before touching `config/ablation/`.
4. **What happens to a tracklet that never gets a gate-passing face?** Today it stays "Detecting..." forever. Should the GIS see it at all? *Leaning: yes, for counting; never for identity.*
5. **Cross-portal (non-adjacent) re-entry** — someone leaves the site and returns hours later. Topology says unreachable; that is wrong. Needs an explicit "exit/re-entry" edge to the outside world.
