#!/usr/bin/env python3
"""
Batched, Vectorized Candidate Blocking Engine using DuckDB.

Same blocking logic (index schema, blocking-key generator, key weights) as
DiskBlockingEngine in sqlite_blocking.py - reused directly from there so the
two engines can never silently diverge. The difference is *how* it's
queried: DiskBlockingEngine looks up one S1 entity at a time (3+ SQLite
round trips per entity, ~5M+ total at full test scale). DuckDBBlockingEngine
instead takes a whole chunk of S1 entities and does each of the 3 lookup
passes (exact name / compressed name / blocking keys) as a single set-based
JOIN across the entire chunk, run by DuckDB's vectorized, multi-threaded
execution engine - collapsing ~5M+ query round trips down to ~3 per chunk.

Meant to be combined with multiprocessing (see pipeline_parallel.py):
each worker process opens its own read-only connection to the same on-disk
DuckDB file (DuckDB supports multiple concurrent read-only connections to
one database file) and calls retrieve_candidates_batch() per chunk.
"""

import csv
import itertools
import os
import time
from typing import Dict, List, Set, Tuple

import duckdb
import pandas as pd
from rapidfuzz import fuzz

from .config import MAX_CANDIDATES_PER_S1, MAX_TOKEN_POSTINGS, RERANK_POOL_SIZE, delimiter_for
from .normalize import normalize_business_name, normalize_address, extract_numbers
from .sqlite_blocking import (
    build_blocking_keys,
    key_hash,
    KEY_WEIGHTS,
    _address_token_candidates,
)
from .transliterate import transliterate_text, load_translit_dict


class DuckDBBlockingEngine:
    def __init__(self, db_path: str = "output/candidates_index.duckdb", max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.db_path = db_path
        self.max_candidates = max_candidates
        self.conn = None

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def build_index(self, s2_path: str, s3_path: str, reset: bool = True,
                     threads: int = None, memory_limit: str = "3GB",
                     translit_dict: Dict[str, str] = None):
        """Ingest S2/S3 candidate files and build the DuckDB candidate index.

        translit_dict: learned Indic-script -> Latin word map (see
        transliterate.py), applied to each candidate's raw name/address
        before normalization so norm_name/norm_addr - and therefore every
        blocking key and feature derived from them - see the transliterated
        text. Defaults to loading data_split/translit_dict.json if present,
        so build_index() is a safe no-op (identical to before this existed)
        when that file hasn't been built yet.
        """
        if translit_dict is None:
            translit_dict = load_translit_dict()
        if reset and os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self.conn = duckdb.connect(self.db_path)
        c = self.conn
        if threads:
            c.execute(f"PRAGMA threads={threads};")
        c.execute(f"PRAGMA memory_limit='{memory_limit}';")

        print(f"Creating tables in {self.db_path}...")
        c.execute("""
            CREATE TABLE candidates (
                rid BIGINT,
                cid VARCHAR,
                name VARCHAR,
                addr VARCHAR,
                country VARCHAR,
                norm_name VARCHAR,
                norm_addr VARCHAR,
                comp_name VARCHAR
            );
        """)

        # Pass 1: ingest + normalize, chunked columnar bulk-append (Arrow/
        # pandas, vectorized) instead of SQLite's row-by-row executemany.
        # Address-token document frequency (needed by build_blocking_keys()
        # to pick rarity-ranked anchors) is accumulated inline here, saving
        # the SQLite engine's separate full read-back scan for the same
        # purpose.
        next_rid = itertools.count()
        df_counter: Dict[Tuple[str, str], int] = {}
        rows_buf: List[tuple] = []
        CHUNK = 200_000

        def flush():
            nonlocal rows_buf
            if not rows_buf:
                return
            df = pd.DataFrame(
                rows_buf,
                columns=["rid", "cid", "name", "addr", "country", "norm_name", "norm_addr", "comp_name"],
            )
            c.append("candidates", df)
            rows_buf = []

        for path in [s2_path, s3_path]:
            if not os.path.exists(path):
                continue
            print(f"Ingesting candidates from: {path}...")
            t0 = time.time()
            count = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter_for(path))
                header = next(reader)
                id_idx = 1 if len(header) > 2 and header[1] == "entity_id" else 0
                name_idx = id_idx + 1
                addr_idx = id_idx + 2
                cntry_idx = id_idx + 3

                for row in reader:
                    if not row or len(row) <= cntry_idx:
                        continue
                    cid = row[id_idx].strip()
                    raw_name = row[name_idx].strip()
                    raw_addr = row[addr_idx].strip()
                    country = row[cntry_idx].strip()

                    translit_name = transliterate_text(raw_name, translit_dict)
                    translit_addr = transliterate_text(raw_addr, translit_dict)
                    norm_name = normalize_business_name(translit_name)
                    comp_name = "".join(norm_name.split()) if len(norm_name.split()) > 1 else ""
                    norm_addr = normalize_address(translit_addr)

                    for w in _address_token_candidates(norm_addr):
                        key = (country, w)
                        df_counter[key] = df_counter.get(key, 0) + 1

                    rows_buf.append((next(next_rid), cid, raw_name, raw_addr, country, norm_name, norm_addr, comp_name))
                    count += 1
                    if len(rows_buf) >= CHUNK:
                        flush()
                        print(f"  Ingested {count} records... ({time.time()-t0:.1f}s)", end="\r")
                flush()

            print(f"\nFinished ingesting {count} records from {os.path.basename(path)} in {time.time()-t0:.1f}s.")

        print("Building indexes on candidates table...")
        t0 = time.time()
        c.execute("CREATE INDEX idx_cand_norm ON candidates (country, norm_name);")
        c.execute("CREATE INDEX idx_cand_comp ON candidates (country, comp_name);")
        c.execute("CREATE UNIQUE INDEX idx_cand_cid ON candidates (cid);")
        print(f"Indexes built in {time.time()-t0:.1f}s.")

        print("Writing address-token document frequency table...")
        t0 = time.time()
        c.execute("CREATE TABLE addr_df (country VARCHAR, token VARCHAR, df INTEGER);")
        if df_counter:
            df_rows = pd.DataFrame(
                [(country, token, df) for (country, token), df in df_counter.items()],
                columns=["country", "token", "df"],
            )
            c.append("addr_df", df_rows)
            del df_rows
        c.execute("CREATE INDEX idx_addr_df ON addr_df (country, token);")
        print(f"addr_df built in {time.time()-t0:.1f}s ({len(df_counter):,} rows).")

        # Pass 2: blocking keys, generated in chunked scans over `candidates`
        # using the exact same key generator DiskBlockingEngine uses at
        # query time (see sqlite_blocking.build_blocking_keys) - guarantees
        # the two engines' notion of "what keys does this record produce"
        # never drifts apart.
        print("Generating blocking keys...")
        t0 = time.time()
        c.execute("CREATE TABLE cand_keys_raw (h BIGINT, rid BIGINT);")
        chunk_size = 200_000
        last_rid = -1
        gen_count = 0
        while True:
            rows = c.execute(
                "SELECT rid, country, norm_name, norm_addr FROM candidates "
                "WHERE rid > ? ORDER BY rid LIMIT ?",
                [last_rid, chunk_size],
            ).fetchall()
            if not rows:
                break
            key_rows = []
            for rid, country, norm_name, norm_addr in rows:
                for key in build_blocking_keys(norm_name, norm_addr, country, df_counter):
                    key_rows.append((key_hash(country, key), rid))
            if key_rows:
                kdf = pd.DataFrame(key_rows, columns=["h", "rid"])
                c.append("cand_keys_raw", kdf)
            last_rid = rows[-1][0]
            gen_count += len(rows)
            print(f"  Keyed {gen_count} records... ({time.time()-t0:.1f}s)", end="\r")
        print(f"\nBlocking keys generated in {time.time()-t0:.1f}s.")
        del df_counter

        # Cap over-frequent postings (near-stopword tokens) with a set-based
        # ANTI JOIN - DuckDB's vectorized/multi-threaded engine runs this as
        # one parallel pass (contrast sqlite_blocking.py's LEFT-JOIN-IS-NULL
        # workaround, needed there because a naive NOT-IN/GROUP-BY hung for
        # 90+ minutes on SQLite's planner).
        print(f"Capping keys with more than {MAX_TOKEN_POSTINGS} postings...")
        t0 = time.time()
        c.execute(
            "CREATE TABLE bad_hashes AS SELECT h FROM cand_keys_raw GROUP BY h HAVING COUNT(*) > ?;",
            [MAX_TOKEN_POSTINGS],
        )
        c.execute(
            "CREATE TABLE cand_keys AS "
            "SELECT ck.h, ck.rid FROM cand_keys_raw ck "
            "ANTI JOIN bad_hashes bh ON ck.h = bh.h;"
        )
        c.execute("CREATE INDEX idx_cand_keys_h ON cand_keys (h);")
        c.execute("DROP TABLE cand_keys_raw;")
        c.execute("DROP TABLE bad_hashes;")
        print(f"Capped in {time.time()-t0:.1f}s.")

        cand_count = c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        key_count = c.execute("SELECT COUNT(*) FROM cand_keys").fetchone()[0]
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        print(
            f"Index built: {cand_count:,} candidates, {key_count:,} keys "
            f"({key_count / max(cand_count, 1):.2f} keys/candidate), "
            f"{db_size / 1e9:.2f} GB on disk."
        )
        c.close()
        self.conn = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def open(self, read_only: bool = True, threads: int = None, memory_limit: str = "1.5GB"):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path, read_only=read_only)
            if threads:
                self.conn.execute(f"PRAGMA threads={threads};")
            self.conn.execute(f"PRAGMA memory_limit='{memory_limit}';")

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get_by_id(self, cid: str):
        """Look up a single candidate record by id - same signature and
        return shape as DiskBlockingEngine.get_by_id() (sqlite_blocking.py):
        (name, addr, country, norm_name, norm_addr) or None. Used by
        train_screener.py/train_tf_specialist.py to inject true-match
        candidates blocking missed, and to look up hard-negative candidates
        by id when mining."""
        if self.conn is None:
            self.open()
        row = self.conn.execute(
            "SELECT name, addr, country, norm_name, norm_addr FROM candidates WHERE cid = ?", [cid]
        ).fetchone()
        return row

    def retrieve_candidates_batch(
        self, s1_chunk: List[Dict[str, str]]
    ) -> Dict[str, List[Tuple[str, str, str, str, str]]]:
        """
        Batched equivalent of DiskBlockingEngine.retrieve_candidates(): takes
        a whole chunk of S1 records and returns {s1_id: [(cid, name, addr,
        norm_name, norm_addr), ...]}. Each of the 3 lookup passes is one
        set-based JOIN across the entire chunk instead of one query per S1
        entity - this is the change that removes the per-entity round-trip
        overhead the original per-record engine paid at full test scale.
        Reranking logic (top RERANK_POOL_SIZE by key-weight, then rapidfuzz
        text similarity, then truncate to max_candidates) is identical to
        DiskBlockingEngine.retrieve_candidates().
        """
        if self.conn is None:
            self.open()
        c = self.conn
        n = len(s1_chunk)
        if n == 0:
            return {}

        s1_rows = []
        for i, s1 in enumerate(s1_chunk):
            norm_name = normalize_business_name(s1["business_name"])
            comp_name = "".join(norm_name.split()) if len(norm_name.split()) > 1 else ""
            norm_addr = normalize_address(s1["business_address"])
            s1_rows.append((i, s1["country"], norm_name, comp_name, norm_addr))

        # Address-token rarity, batched: one lookup covering every distinct
        # (country, token) the whole chunk could query, instead of a
        # separate addr_df round trip per S1 record.
        addr_word_pairs: Set[Tuple[str, str]] = set()
        for _, country, _, _, norm_addr in s1_rows:
            for w in _address_token_candidates(norm_addr):
                addr_word_pairs.add((country, w))

        df_lookup: Dict[Tuple[str, str], int] = {}
        if addr_word_pairs:
            awp_df = pd.DataFrame(list(addr_word_pairs), columns=["country", "token"])
            c.register("awp_df", awp_df)
            for country, token, df in c.execute(
                "SELECT a.country, a.token, ad.df FROM awp_df a "
                "JOIN addr_df ad ON a.country = ad.country AND a.token = ad.token"
            ).fetchall():
                df_lookup[(country, token)] = df
            c.unregister("awp_df")

        # Build every S1 record's blocking keys client-side (same generator
        # as index-build time) and flatten into one (qidx, hash, weight)
        # frame for a single batched join against cand_keys. Weight is
        # resolved from KEY_WEIGHTS here (not in SQL) so there is exactly one
        # place that maps a key prefix to its weight - sqlite_blocking.py's
        # KEY_WEIGHTS dict.
        key_rows = []
        for i, country, norm_name, _, norm_addr in s1_rows:
            for key in build_blocking_keys(norm_name, norm_addr, country, df_lookup):
                prefix = key.split("_", 1)[0] + "_"
                key_rows.append((i, key_hash(country, key), KEY_WEIGHTS.get(prefix, 2.0)))

        # qidx -> {cid: [name, addr, norm_name, norm_addr, score]}
        cand_map_per_q: List[Dict[str, list]] = [dict() for _ in range(n)]

        def _accumulate(qidx, cid, name, addr, c_norm_name, c_norm_addr, weight):
            cm = cand_map_per_q[qidx]
            if cid in cm:
                cm[cid][4] += weight
            else:
                cm[cid] = [name, addr, c_norm_name, c_norm_addr, weight]

        s1_df = pd.DataFrame(s1_rows, columns=["qidx", "country", "norm_name", "comp_name", "norm_addr"])
        c.register("s1_df", s1_df)

        # 1. Exact normalized name (weight 20, capped at 30/entity - mirrors
        #    DiskBlockingEngine's per-entity `LIMIT 30`).
        res = c.execute(
            "SELECT s.qidx, c.cid, c.name, c.addr, c.norm_name, c.norm_addr "
            "FROM s1_df s JOIN candidates c ON s.country = c.country AND s.norm_name = c.norm_name "
            "WHERE s.norm_name != '' "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY s.qidx ORDER BY c.cid) <= 30"
        ).fetchall()
        for qidx, cid, name, addr, c_norm_name, c_norm_addr in res:
            _accumulate(qidx, cid, name, addr, c_norm_name, c_norm_addr, 20.0)

        # 2. Compressed name (no spaces), weight 15, same cap.
        res = c.execute(
            "SELECT s.qidx, c.cid, c.name, c.addr, c.norm_name, c.norm_addr "
            "FROM s1_df s JOIN candidates c ON s.country = c.country AND s.comp_name = c.comp_name "
            "WHERE s.comp_name != '' "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY s.qidx ORDER BY c.cid) <= 30"
        ).fetchall()
        for qidx, cid, name, addr, c_norm_name, c_norm_addr in res:
            _accumulate(qidx, cid, name, addr, c_norm_name, c_norm_addr, 15.0)

        c.unregister("s1_df")

        # 3. Blocking-key matches (tokens, address numbers, zip codes,
        #    bigrams, name-token pairs, cross-script skeletons) - one batched
        #    join for the whole chunk's keys, with the per-key weight SUMmed
        #    *inside* DuckDB's GROUP BY (vectorized, in-database) instead of
        #    fetched as one row per individual key match and summed in a
        #    Python loop. This matters a lot at scale: a chunk can produce
        #    up to MAX_TOKEN_POSTINGS (150) postings per key x ~9 keys per
        #    S1 record x chunk_size records of raw (qidx, cid) rows before
        #    aggregation - millions of rows for an 8,000-record chunk in the
        #    worst case, which measured 3.6GB+ RSS for a single worker
        #    process and OOM-killed 2 of 3 workers on this box's 7.4GB RAM
        #    in the first full-scale run. Aggregating in SQL bounds the
        #    result to at most one row per (qidx, cid) pair instead.
        if key_rows:
            k_df = pd.DataFrame(key_rows, columns=["qidx", "h", "weight"])
            c.register("k_df", k_df)
            res = c.execute(
                "SELECT k.qidx, c.cid, ANY_VALUE(c.name), ANY_VALUE(c.addr), "
                "ANY_VALUE(c.norm_name), ANY_VALUE(c.norm_addr), SUM(k.weight) "
                "FROM k_df k JOIN cand_keys ck ON k.h = ck.h JOIN candidates c ON ck.rid = c.rid "
                "GROUP BY k.qidx, c.cid"
            ).fetchall()
            c.unregister("k_df")
            for qidx, cid, name, addr, c_norm_name, c_norm_addr, weight in res:
                _accumulate(qidx, cid, name, addr, c_norm_name, c_norm_addr, weight)

        # Rerank each S1's pool by text similarity before truncating -
        # identical logic (and RERANK_POOL_SIZE/max_candidates constants) to
        # DiskBlockingEngine.retrieve_candidates().
        out: Dict[str, List[Tuple[str, str, str, str, str]]] = {}
        for i, s1 in enumerate(s1_chunk):
            s1_id = s1["entity_id"]
            cand_map = cand_map_per_q[i]
            if not cand_map:
                out[s1_id] = []
                continue

            norm_name = s1_rows[i][2]
            norm_addr = s1_rows[i][4]
            s1_nums = extract_numbers(norm_addr)

            by_key_weight = sorted(cand_map.items(), key=lambda x: x[1][4], reverse=True)
            pool = by_key_weight[:RERANK_POOL_SIZE]

            def _sim_score(item):
                cid, (name, addr, c_norm_name, c_norm_addr, key_weight) = item
                name_sim = fuzz.token_set_ratio(norm_name, c_norm_name)
                addr_sim = fuzz.token_set_ratio(norm_addr, c_norm_addr)
                num_bonus = 10.0 if (s1_nums & extract_numbers(c_norm_addr)) else 0.0
                return 0.55 * name_sim + 0.35 * addr_sim + num_bonus + 0.01 * key_weight

            sorted_cands = sorted(pool, key=_sim_score, reverse=True)
            out[s1_id] = [
                (cid, data[0], data[1], data[2], data[3])
                for cid, data in sorted_cands[: self.max_candidates]
            ]
        return out


if __name__ == "__main__":
    print("DuckDB blocking engine module ready.")
