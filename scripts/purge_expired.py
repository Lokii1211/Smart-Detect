"""
scripts/purge_expired.py
─────────────────────────
Retention purge job. Deletes identities past their retention date, together
with their embeddings, sightings and snapshot images, and trims the audit log.

Storage limitation is not optional for biometric data: DPDP Act s.8(7)
requires erasure once the purpose is served, and "we kept everything" is the
default failure mode this job exists to prevent.

    python scripts/purge_expired.py                 # DRY RUN (default)
    python scripts/purge_expired.py --execute       # actually delete
    python scripts/purge_expired.py --status        # policy + population only
    python scripts/purge_expired.py --json          # machine-readable

Schedule it. Daily at a quiet hour is usually right:

    # crontab -e
    15 3 * * *  cd /path/to/Smart-Detect-main && \
                .venv/bin/python scripts/purge_expired.py --execute >> logs/purge.log 2>&1

A retention policy that is documented but never runs is worse than none — it
is a false assurance. Verify with --status that the numbers actually move.
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="perform deletions (default is a dry run)")
    ap.add_argument("--status", action="store_true",
                    help="show policy and population, delete nothing")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--actor", default="retention-job",
                    help="recorded in the audit log and erasure receipts")
    args = ap.parse_args()

    from database.db import SessionLocal, init_db
    from backend import governance as gov

    init_db()
    db = SessionLocal()
    try:
        if args.status:
            st = gov.retention_status(db)
            if args.json:
                print(_json.dumps(st, indent=2)); return
            print("Retention policy (days)")
            for k, v in st["policy"].items():
                print(f"  {k:18} {v}")
            print(f"\nIdentities: {st['total']}")
            for k, v in sorted(st["by_consent"].items()):
                print(f"  {k:12} {v}")
            print(f"  legal hold   {st['legal_hold']}")
            print(f"\nPast retention right now: {st['expired_now']}")
            if st["expired_now"]:
                print("  -> run with --execute to erase them")
            return

        res = gov.purge(db, actor=args.actor, dry_run=not args.execute)

        if args.json:
            print(_json.dumps(res, indent=2)); return

        mode = "DRY RUN — nothing deleted" if res["dry_run"] else "EXECUTED"
        print(f"Retention purge [{mode}]  {res['ran_at']}")
        print(f"Policy: {res['policy']}")
        print(f"Candidates past retention: {res['candidates']}")
        for p in res["persons"]:
            print(f"  {p['unique_code']:10} consent={p['consent_status']:10} "
                  f"last_seen={p['last_seen_at'] or '-'} "
                  f"retain_until={p['retain_until'] or '-'} "
                  f"snapshots={p['snapshots']}")
        if not res["dry_run"]:
            ok = sum(1 for e in res["erased"] if e["verified"])
            print(f"\nErased: {len(res['erased'])} ({ok} verified complete)")
            for e in res["erased"]:
                mark = "OK " if e["verified"] else "PARTIAL"
                print(f"  [{mark}] {e['unique_code']}: {e['sightings_deleted']} sightings, "
                      f"{e['snapshots_deleted']} snapshots, receipt {e['receipt_id'][:8]}")
            if res["failed"]:
                print(f"\nFAILED: {len(res['failed'])}")
                for f in res["failed"]:
                    print(f"  {f['unique_code']}: {f['error']}")
            print(f"Audit rows purged: {res['audit_rows_purged']}")
        else:
            print("\nRe-run with --execute to apply.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
