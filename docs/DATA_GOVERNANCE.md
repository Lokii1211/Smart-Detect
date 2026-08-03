# Data Governance — Biometric Data in SmartDetect

**Status:** implemented and tested (`tests/test_governance.py`, 31 tests)
**Last updated:** 2026-08-01

> ### ⚠️ This is not legal advice
> This document is an *engineering* mapping between SmartDetect's mechanisms
> and the obligations they are intended to support. It was written by the
> engineering side, not by a lawyer, and it has not been reviewed by counsel
> or by a Data Protection Officer.
>
> **Before processing any real person's face, have a qualified practitioner
> in your jurisdiction review both this document and the actual deployment.**
> Several obligations below (notably Significant Data Fiduciary status, DPIAs,
> and cross-border transfer) depend on facts about your organisation that this
> repository cannot know.

---

## 1. What data this system holds, and why it is sensitive

| Data | Where | Sensitivity |
|---|---|---|
| **Face embedding** (512-d ArcFace vector) | `persons.face_embedding`, `persons.face_templates` | **Biometric.** Uniquely identifies a person and cannot be reissued — a leaked face is leaked permanently. |
| **Photographs** | `snapshots/<SDT-code>/*.jpg` | Directly identifying image of a real person. |
| **Body re-ID vector, clothing colour** | `persons.reid_embedding`, `dress_color_hsv` | Behavioural/appearance data; identifying in combination. |
| **Movement trail** | `sightings` (camera, zone, timestamp) | Location history — reveals routine, associations, presence at sensitive places. |
| **Audit log** | `audit_log` | Personal data about *both* the subject and the operator. |

A face embedding is not "just numbers." Under **DPDP Act 2023 s.2(t)** it is
personal data; under **GDPR Art. 9(1)** biometric data used for unique
identification is a *special category*; under **Illinois BIPA** a "face
geometry scan" requires written release before collection.

**The movement trail is often more intrusive than the face itself.** A face
embedding says who; the trail says where, when, how often, and with whom.

---

## 2. Lawful basis — the consent register

Every identity carries a `consent_status`. There is no "unset": absence of a
recorded basis is itself a recorded state, and it is treated as the weakest.

| Status | Meaning | Retention (default) | Lawful basis |
|---|---|---|---|
| `consented` | A named individual signed the consent form; `consent_ref` identifies it | 365 days | DPDP s.6 consent; GDPR Art. 9(2)(a) explicit consent |
| `dataset` | Licensed research corpus (e.g. ChokePoint) collected under its own ethics approval | 3650 days | Terms of the dataset licence — **verify it permits your use** |
| `unknown` | **Default.** Auto-registered from a live camera. Nobody agreed to anything | **7 days** | ⚠️ **None established** |

### The `unknown` problem — read this

SmartDetect auto-registers anyone whose face passes the quality gate. Those
people have not consented, were probably not notified, and often do not know
the system exists.

**There is no lawful basis for that under DPDP consent, and "legitimate
interest" is not available for biometrics under GDPR Art. 9.** The short
7-day retention limits the harm; it does not create a lawful basis.

Practically, this means:

- **A pilot with volunteers** — obtain signed forms, set `consented`, and
  operate only in a controlled space. This is the supported path.
- **A public deployment** — you need a specific statutory basis, signage,
  a DPIA, and almost certainly legal counsel. SmartDetect does not make that
  lawful by itself.
- **Bystanders** in a volunteer pilot will still be registered as `unknown`
  if they walk past a camera. Purge frequently, and place cameras so that
  passers-by are out of frame.

Set consent via `PUT /persons/{code}/consent`:

```bash
curl -X PUT "$API/persons/SDT-0003/consent" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"consent_status":"consented","consent_ref":"FORM-2026-001"}'
```

---

## 3. Storage limitation — retention and purge

DPDP s.8(7) requires erasure once the purpose is served. Retention is derived
from consent status, not from one global TTL, because a volunteer and a
stranger cannot lawfully be kept for the same period.

```bash
SMARTDETECT_RETAIN_DAYS_UNKNOWN=7        # no lawful basis -> shortest
SMARTDETECT_RETAIN_DAYS_CONSENTED=365
SMARTDETECT_RETAIN_DAYS_DATASET=3650
SMARTDETECT_RETAIN_DAYS_AUDIT=730        # the audit log is personal data too
```

`retain_until` is computed from `last_seen_at`, so an actively-seen person
does not expire mid-visit. `legal_hold = true` blocks purging (investigation,
litigation hold) and must be set and cleared deliberately.

### The purge job must actually run

```bash
python scripts/purge_expired.py --status     # what is expiring
python scripts/purge_expired.py              # dry run (default)
python scripts/purge_expired.py --execute    # delete

# crontab
15 3 * * *  cd /path/to/Smart-Detect-main && \
            .venv/bin/python scripts/purge_expired.py --execute >> logs/purge.log 2>&1
```

> **A documented retention policy that never executes is worse than none** —
> it is a false assurance to regulators and data subjects. Verify with
> `--status` that the expired count actually falls after a scheduled run.

---

## 4. Data-subject rights (DPDP Act Chapter III)

| Right | Section | Mechanism | Status |
|---|---|---|---|
| Access — what do you hold about me? | s.11 | `GET /persons/{code}` + `GET /person/{code}/trail` | ✅ Implemented |
| Correction | s.12(1) | `PUT /persons/{code}` (name, type) | ⚠️ Partial — an embedding cannot meaningfully be "corrected"; erase and re-enrol |
| **Erasure** | s.12(3), s.8(7) | `DELETE /persons/{code}` | ✅ Implemented, verified |
| Know who accessed my data | s.11(1)(b) | `GET /governance/audit?subject_code=SDT-XXXX` | ✅ Implemented |
| Withdraw consent | s.6(6) | `PUT /persons/{code}/consent` → `unknown`, shortening retention | ✅ Implemented |
| Grievance redressal | s.13 | ❌ **Not implemented** — you must publish a contact route | ❌ **Gap** |
| Nominate (act on subject's behalf) | s.14 | ❌ Not implemented | ❌ Gap |

### Erasure is verified, not assumed

`DELETE /persons/{code}` removes the person row (all embeddings), every
sighting, and the whole `snapshots/<code>/` directory — then **re-queries the
database and re-stats the filesystem** and records the outcome:

```json
{
  "unique_code": "SDT-0011",
  "sightings_deleted": 2, "snapshots_deleted": 3, "embeddings_cleared": 4,
  "verified": true,
  "receipt_id": "05d3ecc1-…"
}
```

`verified: true` means checked, not hoped. The `ErasureReceipt` **survives the
person** and contains no biometric data — it is how you evidence that a
request was honoured. Retrieve via `GET /governance/erasures`.

**Identity codes are not reused.** `SDT-0011` is never reissued after erasure,
so a stale reference cannot silently resolve to a different human.

---

## 5. Accountability — the audit log

Every access to biometric data is recorded: who, whom, when, from where, and
the outcome.

| Action | Logged when |
|---|---|
| `search.by_photo` | Any photo search — **including no-match**, because the pattern of who is searched for is itself what needs oversight |
| `person.read` | Reading one identity's record |
| `person.trail` | Reading movement history |
| `person.erase` | Erasure (and refusals: `not_found`, `denied_legal_hold`) |
| `person.consent_update` | Consent status change |
| `purge.run` | Retention job execution |

Admin-only (`GET /governance/audit`) because the log links operators to the
people they looked up. It carries its own retention (`audit_days`).

**Known weakness:** the log is an ordinary table. A database administrator can
alter it. Genuine tamper-evidence needs append-only storage or hash chaining —
not implemented.

---

## 6. Security safeguards (DPDP s.8(5))

Implemented — see `docs/DEPLOYMENT_TLS.md`:

- Default-deny authentication; 2 public routes out of 42
- Snapshots and API docs require authentication
- Stream tokens scoped so they cannot read the JSON API
- Rate limits on enumeration and search
- No hardcoded credentials; startup fails without configured secrets
- TLS termination documented and required

---

## 7. Breach notification (DPDP s.8(6))

**Not implemented.** The Act requires notifying the Data Protection Board and
each affected Data Principal — with no materiality threshold.

You need, before deployment: a defined detection path, a notification
template, a contact route for every `consented` subject, and an owner. The
audit log supports the forensics; it does not discharge the obligation.

---

## 8. Honest gap list

| Gap | Impact | Severity |
|---|---|---|
| No lawful basis for `unknown` identities | Auto-registration of non-consenting people | 🔴 **Blocking for public deployment** |
| No grievance officer / contact route | DPDP s.13 unmet | 🔴 Blocking |
| No breach notification process | DPDP s.8(6) unmet | 🔴 Blocking |
| No DPIA | Likely required for biometric surveillance | 🔴 Blocking |
| Audit log is not tamper-evident | A DBA can rewrite history | 🟠 High |
| Two shared accounts (`admin`/`operator`) | Cannot attribute a search to an individual — undermines the audit log's purpose | 🟠 High |
| No notice/signage mechanism | Subjects unaware | 🟠 High |
| Embeddings unencrypted at rest | DB file theft = biometric theft | 🟠 High |
| No automated subject-access export | s.11 served manually | 🟡 Medium |
| Backups not covered by purge | Erased data may persist in backups | 🟡 Medium |

**The two shared accounts are the sharpest engineering gap**: an audit log
that can only ever say "operator" cannot answer "which person did this",
which is the question that matters after a misuse complaint.

---

## 9. Deployment checklist

Before any real person's face is processed:

- [ ] Legal review of this document and the deployment by qualified counsel
- [ ] DPIA completed
- [ ] Lawful basis identified and documented for **every** consent class
- [ ] Consent forms signed and filed; `consent_ref` recorded per person
- [ ] Grievance officer appointed and published
- [ ] Breach notification process written, with an owner
- [ ] Retention values set and `purge_expired.py --execute` **scheduled and verified**
- [ ] Per-user accounts replacing the two shared ones
- [ ] Cameras positioned so non-participants are out of frame
- [ ] Signage at every camera
- [ ] Backup retention aligned with the purge policy
- [ ] `python scripts/audit_routes.py --check` passes
- [ ] TLS in place (`docs/DEPLOYMENT_TLS.md`)

---

## 10. Reference

**Migration for existing databases** (adds governance columns; SQLAlchemy
does not alter existing tables):

```bash
python scripts/migrate_governance.py --check   # report
python scripts/migrate_governance.py           # apply
```

Existing rows are backfilled as `unknown` — the defensible default, since no
consent was recorded for them. **They may become immediately expired.** Run
`purge_expired.py --status` right after migrating and record consent for
anyone who should be kept *before* purging.

**Endpoints**

| Method | Path | Role |
|---|---|---|
| `DELETE` | `/persons/{code}` | admin |
| `PUT` | `/persons/{code}/consent` | operator |
| `GET` | `/governance/retention` | operator |
| `POST` | `/governance/purge?dry_run=false` | admin |
| `GET` | `/governance/audit` | admin |
| `GET` | `/governance/erasures` | admin |

**Related:** `docs/CONSENT_FORM_TEMPLATE.md` · `docs/DEPLOYMENT_TLS.md` ·
`PROJECT_REVIEW.md` §4.2 · `backend/governance.py` · `tests/test_governance.py`
