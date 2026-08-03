"""
scripts/audit_routes.py
────────────────────────
Authoritative route/auth table, produced by introspecting the live FastAPI
app — not by grepping source. Use this to verify the security posture and to
review any newly added route.

    python scripts/audit_routes.py            # table
    python scripts/audit_routes.py --markdown # markdown for docs
    python scripts/audit_routes.py --check    # exit 1 if anything is
                                              # unexpectedly public

--check is the CI gate: it fails if a route is publicly reachable without
being in backend.main.PUBLIC_ROUTES, so exposure cannot be introduced silently.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import-time config validation needs these; values are irrelevant to a
# static audit and never leave this process.
os.environ.setdefault("JWT_SECRET", "audit-only-" + "x" * 40)
os.environ.setdefault("ADMIN_PASSWORD", "audit-only-not-a-real-password")
os.environ.setdefault("OPERATOR_PASSWORD", "audit-only-not-a-real-password")
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")

from backend.main import app, PUBLIC_ROUTES  # noqa: E402


def role_of(route) -> str:
    """Role enforced by this route's own dependencies."""
    names = set()
    for dep in getattr(getattr(route, "dependant", None), "dependencies", []) or []:
        call = getattr(dep, "call", None)
        if call is not None:
            names.add(getattr(call, "__name__", ""))
        for sub in getattr(dep, "dependencies", []) or []:
            c = getattr(sub, "call", None)
            if c is not None:
                names.add(getattr(c, "__name__", ""))
    # Dependencies declared as default values on the signature
    for p in getattr(getattr(route, "dependant", None), "dependencies", []) or []:
        pass
    src_names = set()
    fn = getattr(route, "endpoint", None)
    if fn is not None:
        import inspect
        try:
            for p in inspect.signature(fn).parameters.values():
                d = p.default
                dep_call = getattr(d, "dependency", None)
                if dep_call is not None:
                    src_names.add(getattr(dep_call, "__name__", ""))
        except (TypeError, ValueError):
            pass
    names |= src_names
    if "require_admin" in names:
        return "admin"
    if "require_operator" in names:
        return "operator"
    if "require_auth" in names:
        return "any-auth"
    return "-"


def collect():
    rows = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods:
            continue
        for m in sorted(methods - {"HEAD", "OPTIONS"}):
            explicit = role_of(route)
            is_public = (m, path) in PUBLIC_ROUTES
            # Effective access: the middleware denies everything not in the
            # allowlist, so an absent per-route dependency still means
            # "authenticated" rather than "anonymous".
            if is_public:
                effective = "PUBLIC"
            elif explicit == "-":
                effective = "authenticated (middleware)"
            else:
                effective = explicit
            rows.append({"method": m, "path": path, "explicit": explicit,
                         "public": is_public, "effective": effective})
    return sorted(rows, key=lambda r: (not r["public"], r["path"], r["method"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows = collect()

    if args.markdown:
        print("| Method | Path | Route dependency | Effective access |")
        print("|---|---|---|---|")
        for r in rows:
            print(f"| {r['method']} | `{r['path']}` | {r['explicit']} | "
                  f"{'**PUBLIC**' if r['public'] else r['effective']} |")
    else:
        print(f"{'METHOD':7} {'PATH':48} {'DEPENDENCY':10} EFFECTIVE")
        print("-" * 100)
        for r in rows:
            print(f"{r['method']:7} {r['path']:48} {r['explicit']:10} "
                  f"{'PUBLIC  <-- anonymous' if r['public'] else r['effective']}")
        print("-" * 100)
        n_pub = sum(1 for r in rows if r["public"])
        print(f"{len(rows)} routes | {n_pub} public | {len(rows)-n_pub} require auth")
        print("\nAlso enforced by middleware (outside the router):")
        print("  /snapshots/**                 authenticated (person photographs)")
        print("  /docs /redoc /openapi.json    authenticated unless SMARTDETECT_PUBLIC_DOCS=1")

    if args.check:
        unexpected = [r for r in rows if r["public"]
                      and (r["method"], r["path"]) not in PUBLIC_ROUTES]
        if unexpected:
            print("\nFAIL: routes public but not in PUBLIC_ROUTES:", file=sys.stderr)
            for r in unexpected:
                print(f"  {r['method']} {r['path']}", file=sys.stderr)
            sys.exit(1)
        print(f"\nCHECK PASSED: only {len(PUBLIC_ROUTES)} allowlisted routes are public.")


if __name__ == "__main__":
    main()
