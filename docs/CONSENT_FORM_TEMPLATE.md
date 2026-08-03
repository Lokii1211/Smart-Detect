# Consent Form — Template for Volunteer Recording

> ### ⚠️ TEMPLATE — must be reviewed before use
> This is a **starting draft written by engineers, not lawyers**. It has not
> been reviewed by counsel, an ethics committee, or a Data Protection Officer.
>
> Before giving this to a single volunteer:
> 1. Have it reviewed by a qualified practitioner in your jurisdiction.
> 2. If this is academic work, obtain **IRB / institutional ethics approval**
>    first — most institutions require approval *before* recruitment, and
>    retroactive approval is usually not possible.
> 3. Replace every `[BRACKETED]` field. A form with placeholders left in is
>    not valid consent.
>
> Consent must be **free, specific, informed, unconditional and unambiguous**
> (DPDP Act s.6(1)). Consent obtained by pressure — from a supervisor,
> lecturer, or employer — is not free, and a court is likely to say so.

---

## Notes for whoever runs the session

**Power imbalance is the most common way this goes wrong.** If you are
recruiting your own students, employees, or team members, they may not feel
able to refuse. Mitigations: recruit through a neutral party, make
non-participation genuinely consequence-free and *say so*, and avoid
recruiting people who report to you.

**Bystanders.** Anyone who walks into frame is auto-registered as
`consent_status='unknown'` with no lawful basis. Position cameras so
non-participants cannot be captured, use a closed room, and purge after each
session:

```bash
python scripts/purge_expired.py --status
python scripts/purge_expired.py --execute
```

**Under-18s.** DPDP s.9 requires verifiable parental consent and prohibits
tracking and behavioural monitoring of children. **Do not enrol minors with
this form.** That needs a separate process and separate approval.

**After each enrolment,** record the basis in the system so retention and
audit reflect reality:

```bash
curl -X PUT "$API/persons/SDT-0042/consent" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"consent_status":"consented","consent_ref":"SD-2026-0042"}'
```

File the signed paper/PDF against that same `consent_ref`. A `consent_ref`
pointing at nothing is worse than none — it looks like evidence and is not.

---
---

# CONSENT TO COLLECTION AND PROCESSING OF BIOMETRIC DATA

**Project:** [PROJECT NAME]
**Organisation:** [ORGANISATION / INSTITUTION]
**Data Fiduciary:** [LEGAL ENTITY NAME]
**Ethics / IRB approval reference:** [REFERENCE — or state "not applicable" and why]
**Consent reference:** `SD-________`  *(office use — matches the system record)*

---

## 1. What we are asking

We are asking to record video of you and to use it to test a person-recognition
system. If you agree, we will record you walking through, or standing in front
of, one or more cameras at [LOCATION].

**Taking part is entirely voluntary. If you say no, nothing happens — there is
no penalty, and it will not affect your studies, employment, grades, or
standing in any way.**

## 2. What we will collect and keep

If you take part, the system will store:

1. **A mathematical description of your face** (a "face embedding"). This is a
   list of numbers derived from your facial features. It is **biometric data**.
   It is used to recognise you again.
2. **Photographs of you**, cropped from the video.
3. **A description of your body and clothing** used to re-identify you when
   your face is not visible.
4. **A record of where and when each camera saw you** — camera, zone, and
   timestamp.
5. **An identifier** (`SDT-XXXX`) linking these together, plus your name if you
   provide it.

### Please understand this clearly

**Your face cannot be changed if it is ever leaked.** Unlike a password, you
cannot be issued a new one. This is the main reason we are asking so formally.

## 3. Why we are collecting it

[STATE THE SPECIFIC PURPOSE — e.g. "to measure how accurately the system
tells different people apart under varying lighting and camera angles."]

We will **not** use your data for:

- Any purpose other than the one stated above
- Real security, policing, employment, or access-control decisions about you
- Training or fine-tuning a model that is published or shared outside
  [ORGANISATION]
- Any commercial product
- Sharing or sale to any third party

## 4. How long we keep it

Your data is deleted **[N] days after the last recording**, or immediately on
request — whichever comes first.

The system enforces this automatically. After deletion we retain only a
**deletion receipt**: your identifier, the date, and counts of what was
removed. It contains **no image of you and no biometric data**, and exists so
we can prove your data was actually deleted.

## 5. Who can see it

- Named researchers/operators on this project: [NAMES OR ROLES]
- Stored on [WHERE — e.g. "a single encrypted machine at X, not on the internet"]
- **Never** published, shared, or transferred outside [ORGANISATION] without
  separate written consent from you
- Every access is logged. You can ask for that log — see §6.

## 6. Your rights

You may, at any time and without giving a reason:

| Right | How |
|---|---|
| **Withdraw consent** | Contact us. Recording stops and data is deleted. |
| **Have your data deleted** | Contact us. Deletion is verified and you receive the receipt. |
| **See what we hold** | We will show you your record and photographs. |
| **See who accessed it** | We will provide the access log for your identifier. |
| **Correct your details** | Contact us (name, contact details). |
| **Complain** | To our grievance officer (§8), or to the Data Protection Board of India. |

**Withdrawing consent will not disadvantage you in any way.**

Withdrawal is not retroactive for analysis already published in aggregate,
statistical form that cannot identify you — but your underlying data is still
deleted.

## 7. Risks

We are telling you these plainly rather than reassuring you:

- **Irreversibility.** A leaked face embedding cannot be reissued.
- **Breach.** We use access control, encryption in transit, and audit logging,
  but no system is perfectly secure.
- **Re-identification.** Even data with your name removed may identify you —
  it is a picture of your face.
- **Function creep.** We commit in writing above to the stated purpose only.

## 8. Contact

| | |
|---|---|
| Researcher / operator | [NAME], [EMAIL], [PHONE] |
| Supervisor / responsible person | [NAME], [EMAIL] |
| **Grievance officer** (DPDP s.13) | [NAME], [EMAIL] |
| Ethics committee / IRB | [CONTACT] |
| Regulator | Data Protection Board of India |

---

## 9. Declaration

Please initial each box. **Do not initial anything that is not true.**

| | Initial |
|---|---|
| I have read and understood this form and had the chance to ask questions. | ☐ |
| I understand my **face embedding is biometric data** and that my face cannot be reissued if leaked. | ☐ |
| I understand taking part is **voluntary** and I can withdraw at any time with **no penalty**. | ☐ |
| I understand what is collected, why, how long it is kept, and who can see it. | ☐ |
| I understand how to withdraw consent and request deletion. | ☐ |
| I am **18 or older**. | ☐ |
| **I consent** to the collection and processing of my biometric data as described. | ☐ |

**Optional — tick only if you agree. Declining does not affect participation:**

| | Tick |
|---|---|
| My photographs may be used in internal presentations to [ORGANISATION] staff. | ☐ |
| I may be contacted about future related studies. | ☐ |

---

**Participant**

Full name: ______________________________________

Signature: ______________________________  Date: ______________

Contact (for deletion requests / notification): ______________________________

**Person taking consent** — I have explained this study and believe the
participant understands it and consents freely.

Name: ______________________________________

Signature: ______________________________  Date: ______________

---

*Two copies: one for the participant, one filed against consent reference
`SD-________`. Give the participant their copy at the time of signing.*

---
---

## Withdrawal of consent

*(Detach or submit separately. Accept withdrawal by any means — email,
verbally, this form. Do not require a specific format.)*

I, ______________________________ (consent ref `SD-________`), withdraw my
consent and request deletion of all my data.

Signature: ______________________________  Date: ______________

**For the operator — complete on receipt:**

```bash
curl -X DELETE "$API/persons/SDT-XXXX?reason=consent_withdrawn" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

- Erasure receipt ID: ______________________  `verified: true` ☐
- Deletion confirmed to participant on: ______________ by: ______________
- Backups containing this data purged/scheduled: ______________

> Deletion must be actioned **without undue delay**. If backups still contain
> the data, say so honestly to the participant and give the date they expire.
