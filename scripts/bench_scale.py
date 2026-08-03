"""
scripts/bench_scale.py
───────────────────────
Measurements that ground docs/SCALE_ARCHITECTURE.md. Nothing here changes
production code — it exercises the existing paths and reports numbers.

Three benchmarks:

  identity-search   _find_person_python (current, linear scan) vs pgvector
                    exact vs pgvector HNSW vs in-process hnswlib,
                    at 1k / 10k / 100k identities.

  pgvector-path     Executes the never-run _find_person_pgvector against a
                    real PostgreSQL+pgvector, verifying it returns the same
                    match as the SQLite path.

  writers           Concurrent sighting writers against SQLite vs PostgreSQL,
                    which is the multi-camera contention question.

    export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
    python scripts/bench_scale.py --pg "postgresql://$USER@localhost/smartdetect_scale" --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SMARTDETECT_NO_AUTOSTART", "1")
os.environ.setdefault("JWT_SECRET", "bench-only-" + "x" * 40)
os.environ.setdefault("ADMIN_PASSWORD", "bench-only-not-real")
os.environ.setdefault("OPERATOR_PASSWORD", "bench-only-not-real")

import numpy as np  # noqa: E402

DIM = 512
SIZES = (1_000, 10_000, 100_000)


def unit_rows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def probe_queries(vecs: np.ndarray, k: int, sim: float = 0.85,
                  seed: int = 3) -> tuple:
    """
    Realistic probes: each query is a stored vector perturbed to ~`sim` cosine.
    This is what face matching actually does (a new view of a known person).
    Random-vs-random probes make every candidate near-orthogonal, which is the
    worst case for ANN recall and is NOT representative.
    Returns (queries, ground_truth_indices).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(vecs.shape[0], size=k, replace=False)
    base = vecs[idx]
    noise = rng.standard_normal((k, DIM)).astype(np.float32)
    noise -= (np.sum(noise * base, axis=1, keepdims=True)) * base
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    q = sim * base + np.sqrt(max(0.0, 1 - sim * sim)) * noise
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32), idx


def timed(fn, runs: int) -> float:
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs


# ── 1. Current implementation: Python linear scan ───────────────────────────

def bench_python_scan(vecs: np.ndarray, queries: np.ndarray, runs: int) -> float:
    """
    Mirrors database.queries._find_person_python: pull every row, json.loads
    each vector, cosine against the query in a Python loop, track the best.
    Reproduced here (not imported) so it can be measured without a populated
    database at 100k scale; the arithmetic is identical.
    """
    stored = [v for v in vecs]        # already decoded — favours the current impl

    def one():
        q = queries[0]
        qn = float(np.linalg.norm(q)) + 1e-8
        best, best_sim = None, -1.0
        for i, s in enumerate(stored):
            sim = float(np.dot(q, s)) / (qn * (float(np.linalg.norm(s)) + 1e-8))
            if sim > best_sim:
                best_sim, best = sim, i
        return best
    return timed(one, runs)


def bench_numpy_matmul(vecs: np.ndarray, queries: np.ndarray, runs: int) -> float:
    """Same exhaustive search, vectorised — the cheapest possible fix."""
    def one():
        return int(np.argmax(vecs @ queries[0]))
    return timed(one, runs)


# ── 2. hnswlib (in-process ANN) ─────────────────────────────────────────────

def bench_hnswlib(vecs: np.ndarray, queries: np.ndarray, runs: int, gt_idx=None):
    import hnswlib
    n = vecs.shape[0]
    idx = hnswlib.Index(space="cosine", dim=DIM)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    t0 = time.perf_counter()
    idx.add_items(vecs, np.arange(n))
    build_s = time.perf_counter() - t0
    idx.set_ef(64)

    def one():
        return idx.knn_query(queries[0], k=1)
    q = timed(one, runs)

    # recall@1 vs exhaustive ground truth (the true nearest stored vector)
    gt = gt_idx if gt_idx is not None else np.argmax(vecs @ queries.T, axis=0)
    labels, _ = idx.knn_query(queries, k=1)
    recall = float(np.mean(labels[:, 0] == gt))
    return q, build_s, recall


# ── 3. PostgreSQL + pgvector ────────────────────────────────────────────────

def pg_setup(pg_url: str, vecs: np.ndarray, use_hnsw: bool):
    from sqlalchemy import create_engine, text
    eng = create_engine(pg_url)
    n = vecs.shape[0]
    with eng.begin() as c:
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        c.execute(text("DROP TABLE IF EXISTS persons CASCADE"))
        c.execute(text(f"""
            CREATE TABLE persons (
                id text PRIMARY KEY,
                unique_code varchar(32) UNIQUE NOT NULL,
                face_embedding vector({DIM}),
                reid_embedding vector({DIM}),
                last_seen_at timestamp
            )"""))
    # COPY is the only sane way to load 100k rows
    import io, psycopg2
    raw = pg_url.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(raw)
    buf = io.StringIO()
    for i in range(n):
        vs = "[" + ",".join(f"{x:.6f}" for x in vecs[i]) + "]"
        buf.write(f"{uuid.uuid4()}\tSDT-{i:06d}\t{vs}\t\\N\t2026-08-02 00:00:00\n")
    buf.seek(0)
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.copy_from(buf, "persons",
                      columns=("id", "unique_code", "face_embedding",
                               "reid_embedding", "last_seen_at"))
    conn.commit()
    load_s = time.perf_counter() - t0

    build_s = 0.0
    if use_hnsw:
        t0 = time.perf_counter()
        with eng.begin() as c:
            c.execute(text("SET maintenance_work_mem='512MB'"))
            c.execute(text("CREATE INDEX ON persons "
                           "USING hnsw (face_embedding vector_cosine_ops) "
                           "WITH (m=16, ef_construction=64)"))
        build_s = time.perf_counter() - t0
    with eng.begin() as c:
        c.execute(text("ANALYZE persons"))
    conn.close()
    return eng, load_s, build_s


def bench_pgvector(eng, queries: np.ndarray, runs: int) -> float:
    """Executes the REAL production statement shape from
    database.queries._find_person_pgvector."""
    from sqlalchemy import text
    # NOTE: the column is NOT cast. Wrapping it (CAST(face_embedding AS vector))
    # makes the ORDER BY a non-indexable expression and pgvector falls back to
    # a Seq Scan — verified with EXPLAIN. The parameter is cast instead.
    stmt = text("""
        SELECT unique_code, 1 - (face_embedding <=> CAST(:vec AS vector)) AS similarity
        FROM persons
        WHERE face_embedding IS NOT NULL
        ORDER BY face_embedding <=> CAST(:vec AS vector)
        LIMIT 1
    """)
    vec = "[" + ",".join(f"{x:.6f}" for x in queries[0]) + "]"

    def one():
        with eng.connect() as c:
            c.execute(text("SET hnsw.ef_search = 100"))
            return c.execute(stmt, {"vec": vec}).fetchone()
    return timed(one, runs)


def pg_uses_index(eng, queries) -> bool:
    """EXPLAIN the real query and report whether HNSW is actually used."""
    from sqlalchemy import text
    vec = "[" + ",".join(f"{x:.6f}" for x in queries[0]) + "]"
    with eng.connect() as c:
        c.execute(text("SET hnsw.ef_search = 100"))
        plan = "\n".join(r[0] for r in c.execute(text(
            "EXPLAIN SELECT unique_code FROM persons WHERE face_embedding IS NOT NULL "
            "ORDER BY face_embedding <=> CAST(:vec AS vector) LIMIT 1"), {"vec": vec}))
    return "Index Scan" in plan


def cmd_search(args) -> dict:
    out = {}
    for n in SIZES:
        if n > args.max_size:
            continue
        print(f"\n=== {n:,} identities ===")
        vecs = unit_rows(n, seed=1)
        queries, gt_idx = probe_queries(vecs, 20, sim=0.85, seed=3)
        runs = 5 if n >= 100_000 else 20
        row = {"n": n}

        t = bench_python_scan(vecs, queries, runs)
        row["python_scan_ms"] = 1000 * t
        print(f"  python linear scan (current)   {1000*t:9.2f} ms/query")

        t = bench_numpy_matmul(vecs, queries, runs)
        row["numpy_matmul_ms"] = 1000 * t
        print(f"  numpy matmul (exhaustive)      {1000*t:9.2f} ms/query")

        try:
            t, build, recall = bench_hnswlib(vecs, queries, runs, gt_idx)
            row.update(hnswlib_ms=1000 * t, hnswlib_build_s=build, hnswlib_recall=recall)
            print(f"  hnswlib ANN                    {1000*t:9.2f} ms/query "
                  f"(build {build:.1f}s, recall@1 {recall:.3f})")
        except Exception as e:
            print("  hnswlib failed:", e)

        if args.pg:
            for use_hnsw in (False, True):
                try:
                    eng, load_s, build_s = pg_setup(args.pg, vecs, use_hnsw)
                    t = bench_pgvector(eng, queries, runs)
                    used = pg_uses_index(eng, queries)
                    row[("pgvector_hnsw_index_used" if use_hnsw else "pgvector_exact_index_used")] = used
                    key = "pgvector_hnsw_ms" if use_hnsw else "pgvector_exact_ms"
                    row[key] = 1000 * t
                    if use_hnsw:
                        row["pgvector_hnsw_build_s"] = build_s
                    label = "pgvector + HNSW index" if use_hnsw else "pgvector exact (no index)"
                    extra = (f" (build {build_s:.1f}s, index_used={used})" if use_hnsw
                             else f" (load {load_s:.1f}s)")
                    print(f"  {label:30} {1000*t:9.2f} ms/query{extra}")
                    eng.dispose()
                except Exception as e:
                    print(f"  pgvector ({'hnsw' if use_hnsw else 'exact'}) failed: {e}")
        out[str(n)] = row
    return out


# ── 4. Verify the untested production pgvector code path ────────────────────

def cmd_verify_pgpath(args) -> dict:
    """Run database.queries.find_person_by_embedding against real PostgreSQL
    and confirm it agrees with the SQLite/Python path on the same data."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    import database.queries as Q

    n = 500
    vecs = unit_rows(n, seed=7)
    eng, _, _ = pg_setup(args.pg, vecs, use_hnsw=False)
    Session = sessionmaker(bind=eng)
    db = Session()

    # Query = a stored vector perturbed slightly; the same row must win.
    target = 123
    q = vecs[target] * 0.93 + unit_rows(1, seed=99)[0] * 0.07
    q = q / np.linalg.norm(q)

    res = {"is_postgres_detected": Q._is_postgres(db),
           "expected_code": f"SDT-{target:06d}"}

    # (a) the production statement, exactly as shipped
    try:
        pg_hit = Q.find_person_by_embedding(q, db=db, threshold=0.5)
        res["production_path"] = "ok"
        res["pgvector_result"] = pg_hit
        res["pgvector_correct"] = bool(pg_hit and pg_hit["unique_code"] == f"SDT-{target:06d}")
    except Exception as e:
        db.rollback()
        pg_hit = None
        res["production_path"] = "FAILED"
        res["production_error"] = type(e).__name__ + ": " + str(e).split("\n")[0][:160]
        res["pgvector_correct"] = False

    # (b) the same query with ::vector replaced by CAST(... AS vector).
    #     SQLAlchemy's text() bind-parser mis-reads "::" — it extracts a
    #     parameter named "ve" from ":vec::vector" — so the shipped statement
    #     can never bind. This shows the fix works and how fast it is.
    from sqlalchemy import text as _text
    fixed = _text("""
        SELECT unique_code,
               1 - (CAST(face_embedding AS vector) <=> CAST(:vec AS vector)) AS similarity
        FROM persons
        WHERE face_embedding IS NOT NULL
        ORDER BY CAST(face_embedding AS vector) <=> CAST(:vec AS vector)
        LIMIT 1
    """)
    vecstr = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
    try:
        row = db.execute(fixed, {"vec": vecstr}).fetchone()
        res["fixed_path"] = "ok"
        res["fixed_result"] = {"unique_code": row.unique_code,
                               "similarity": round(float(row.similarity), 4)}
        res["fixed_correct"] = row.unique_code == f"SDT-{target:06d}"
    except Exception as e:
        db.rollback()
        res["fixed_path"] = "FAILED"
        res["fixed_error"] = str(e).split("\n")[0][:160]
        res["fixed_correct"] = False

    # Same maths in the Python path for comparison
    sims = vecs @ q
    res["python_best_code"] = f"SDT-{int(np.argmax(sims)):06d}"
    res["python_best_sim"] = float(np.max(sims))
    res["agree"] = res["python_best_code"] == (pg_hit or {}).get("unique_code")

    db.close(); eng.dispose()
    print("\n=== production pgvector path (first execution) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


# ── 5. Concurrent writers: SQLite vs PostgreSQL ─────────────────────────────

def cmd_writers(args) -> dict:
    """
    Simulates N cameras writing sightings at once — the contention that
    decides whether SQLite survives multi-camera.
    """
    import threading
    from sqlalchemy import create_engine, text

    def run(url: str, label: str, n_writers: int, per_writer: int) -> dict:
        eng = create_engine(url, pool_size=max(n_writers + 2, 5), max_overflow=10) \
            if url.startswith("postgres") else create_engine(
                url, connect_args={"check_same_thread": False})
        with eng.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS bench_sightings"))
            c.execute(text("""CREATE TABLE bench_sightings (
                id text PRIMARY KEY, unique_code text, camera_id text,
                seen_at timestamp, confidence double precision)"""))
        if url.startswith("sqlite"):
            with eng.begin() as c:
                c.execute(text("PRAGMA journal_mode=WAL"))
                c.execute(text("PRAGMA busy_timeout=30000"))

        errors, lat = [], []
        lock = threading.Lock()

        def writer(wid: int):
            local = []
            for i in range(per_writer):
                t0 = time.perf_counter()
                try:
                    with eng.begin() as c:
                        c.execute(text("INSERT INTO bench_sightings VALUES "
                                       "(:i,:u,:c,now_placeholder,:f)".replace(
                                           "now_placeholder", "CURRENT_TIMESTAMP")),
                                  {"i": str(uuid.uuid4()), "u": f"SDT-{wid:04d}",
                                   "c": f"CAM-{wid}", "f": 0.9})
                    local.append(time.perf_counter() - t0)
                except Exception as e:
                    with lock:
                        errors.append(str(e)[:120])
            with lock:
                lat.extend(local)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.perf_counter() - t0
        eng.dispose()

        total = n_writers * per_writer
        r = {"backend": label, "writers": n_writers, "inserts": total,
             "wall_s": wall, "throughput_per_s": (total - len(errors)) / max(wall, 1e-9),
             "errors": len(errors),
             "p50_ms": 1000 * float(np.percentile(lat, 50)) if lat else None,
             "p99_ms": 1000 * float(np.percentile(lat, 99)) if lat else None,
             "sample_error": errors[0] if errors else None}
        print(f"  {label:12} writers={n_writers:3} {r['throughput_per_s']:8.0f} ins/s "
              f"p50={r['p50_ms']:6.2f}ms p99={r['p99_ms']:7.2f}ms errors={r['errors']}")
        return r

    out = []
    tmp_sqlite = f"sqlite:///{ROOT}/eval/results/perf/bench_writers.db"
    Path(f"{ROOT}/eval/results/perf").mkdir(parents=True, exist_ok=True)
    print("\n=== concurrent writers (each insert its own transaction) ===")
    for nw in (1, 4, 8, 16):
        out.append(run(tmp_sqlite, "SQLite/WAL", nw, 100))
    if args.pg:
        for nw in (1, 4, 8, 16):
            out.append(run(args.pg, "PostgreSQL", nw, 100))
    return {"writers": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", default="", help="postgres URL, e.g. postgresql://user@localhost/db")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--verify-pgpath", action="store_true")
    ap.add_argument("--writers", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-size", type=int, default=100_000)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    results = {}
    if args.all or args.search:
        results["search"] = cmd_search(args)
    if (args.all or args.verify_pgpath) and args.pg:
        results["pgpath"] = cmd_verify_pgpath(args)
    if args.all or args.writers:
        results.update(cmd_writers(args))

    if args.json_out:
        p = Path(args.json_out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
