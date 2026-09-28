#!/usr/bin/env python3
"""
High-Performance, Low-Memory Blocking Engine using SQLite on Disk.

Solves the 7.4 GB RAM memory constraint:
- Uses an on-disk SQLite database with B-Tree indexes.
- Peak RAM usage is < 200 MB (instead of 6+ GB in pure Python objects).
- Scales effortlessly to 10+ million candidate records.
- Microsecond query latency per S1 entity.
"""

import csv
import hashlib
import itertools
import os
import sqlite3
import time
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz

from .config import (
    MAX_CANDIDATES_PER_S1,
    MAX_TOKEN_POSTINGS,
    STOPWORDS,
    RERANK_POOL_SIZE,
)
from .transliterate import transliterate_text, load_translit_dict
from .normalize import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    phonetic_skeleton,
)

ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "highway", "court",
    "suite", "apartment", "unit", "floor", "near", "opposite", "infront", "behind",
    "colony", "nagar", "village", "town", "district", "city", "state", "door", "no",
    "rue", "allee", "chemin", "place", "impasse", "france", "india", "delhi", "ny", "ca",
    "tx", "fl", "il", "pa", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma",
}

# Skeletons that are just a legal suffix ("private limited" -> "prvt"/"lmtd")
# survived phonetic_skeleton() folding and would anchor unrelated companies
# together if used as a key on their own.
SKELETON_STOPWORDS = {
    "prvt", "prvat", "pvt", "lmtd", "ltd", "lmted", "prat", "pra",
    "li", "llp", "kmpn", "kompn", "ind", "indstrs", "limitd",
}

# Countries where a Latin/Brahmic script mismatch between S1 and S2/S3 is
# common enough to justify the extra sk_ key (see _skeleton_tokens below).
SKELETON_KEY_COUNTRIES = {"India"}


def _distinctive_tokens(norm_name: str, limit: int = 3) -> List[str]:
    return [t for t in norm_name.split() if len(t) >= 4 and t not in STOPWORDS][:limit]


def _address_token_candidates(norm_addr: str) -> Set[str]:
    """Address words eligible to be a blocking-key anchor token (long enough,
    not a bare number, not a generic address word). Shared between
    _rare_addr_tokens (which ranks these by rarity) and the addr_df
    document-frequency scan / query-time lookup (which need to know exactly
    which words a rarity ranking would ever consider) so the two can't
    silently diverge."""
    return {
        w for w in norm_addr.split()
        if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
    }


def _rare_addr_tokens(
    norm_addr: str,
    country: str,
    df_lookup: Dict[Tuple[str, str], int],
    limit: int = 2,
) -> List[str]:
    """
    Pick the `limit` rarest (lowest document-frequency) address tokens for
    this country, ties broken alphabetically - order-invariant (doesn't
    depend on where in the address the token appears, unlike the old
    first-N-tokens approach) and self-adjusting (a token missing from
    df_lookup, e.g. one that appears only in this record, sorts as df=0 and
    so is preferred, same as the old approach's implicit "first is best"
    guess, but now driven by actual rarity instead of position).
    """
    candidates = _address_token_candidates(norm_addr)
    return sorted(candidates, key=lambda w: (df_lookup.get((country, w), 0), w))[:limit]


def _skeleton_tokens(norm_name: str, limit: int = 2) -> List[str]:
    """
    Phonetic consonant skeletons (see normalize.phonetic_skeleton) of the
    first `limit` name tokens long/distinctive enough to be worth folding.
    Used to bridge a Latin/Brahmic script mismatch between S1 and S2/S3 for
    the *same* business (e.g. "Global Developers" vs "ग्लोबल डेवलपर्स") -
    the skeleton is the only thing the two spellings have in common.
    """
    out: List[str] = []
    for t in norm_name.split():
        if len(t) < 3:
            continue
        sk = phonetic_skeleton(t)
        if len(sk) >= 3 and sk not in SKELETON_STOPWORDS:
            out.append(sk)
        if len(out) >= limit:
            break
    return out


def build_blocking_keys(
    norm_name: str,
    norm_addr: str,
    country: str = "",
    df_lookup: Dict[Tuple[str, str], int] = None,
) -> List[str]:
    """
    Generate blocking keys for a (normalized name, normalized address) pair.

    Used identically at index-build time and at query time, so the keys a
    record is indexed under always match the keys a query can produce for
    the same underlying text (previously these could silently diverge,
    e.g. via Python's per-process set-iteration order on `nums`).

    Keys are order-invariant with respect to address/name word order. This
    also requires the *same* df_lookup on both sides (address-token rarity
    is relative to the whole candidate pool) - both build_index() and
    retrieve_candidates() read it from the same on-disk addr_df table, so
    it can't silently diverge the way an in-process-only stat would.
    """
    df_lookup = df_lookup or {}
    keys: List[str] = []

    # 1. Distinctive name tokens, plus sorted pairs of them. The pair keys
    #    (tp_) recover a match when one of the two names' individual tok_
    #    keys got capped as near-stopword-frequent (e.g. "collins" or
    #    "cornerstone" alone is common) but the specific *pair* is still
    #    selective - measured on a 2,000-entity validation sample, this
    #    alone would recover 6.6% of currently-missed true pairs.
    dt = _distinctive_tokens(norm_name, limit=3)
    for t in dt:
        keys.append(f"tok_{t}")
    for a, b in itertools.combinations(sorted(dt), 2):
        keys.append(f"tp_{a}_{b}")

    nums = sorted(extract_numbers(norm_addr))
    addr_tokens = _rare_addr_tokens(norm_addr, country, df_lookup, limit=2)

    # 2. House/street number + address token: every number (the
    #    order-invariance/multi-number fix), paired with the two rarest
    #    address tokens instead of just one. This was originally cut to one
    #    token (37.1% of cand_keys rows at the old 7.4GB WSL2 RAM cap, in
    #    the pre-hash-compaction TEXT-key format) - restoring it now that
    #    the RAM cap is 11GB and keys are stored as compact int64 hashes.
    #    A full revert of BOTH num_ (2x) and bg_ (4.2x, see below) together
    #    was tried and OOM-killed the build at ~7GB with CREATE INDEX still
    #    ahead of it - num_ alone is a much smaller change (~2x one key
    #    type only) and is kept; bg_ stays at its original single-pair trim.
    for num in nums[:3]:
        for atok in addr_tokens[:2]:
            keys.append(f"num_{num}_{atok}")

    # 3. Postal/PIN-code-shaped numbers (4+ digits) alone are still a
    #    useful anchor even when no address token matches (missing/garbled
    #    street text, landmark-only addresses, etc).
    for num in nums:
        if len(num) >= 4:
            keys.append(f"zip_{num}")

    # 4. Unordered address token pair (order-invariant bigram) of the two
    #    rarest tokens. Reverting this to the full top-4-token combinations
    #    (up to 6 keys/record, ~4.2x growth) was the change that OOM-killed
    #    the build - see the num_ comment above. Kept at the original
    #    single-pair trim.
    if len(addr_tokens) >= 2:
        a, b = sorted(addr_tokens[:2])
        keys.append(f"bg_{a}_{b}")

    # 5. Cross-script name-skeleton key (India only - see
    #    SKELETON_KEY_COUNTRIES). Measured to recover 6.6% of currently-
    #    missed true pairs, almost all of them Latin/Devanagari (or other
    #    Brahmic script) spellings of the same business name.
    if country in SKELETON_KEY_COUNTRIES:
        sk = _skeleton_tokens(norm_name, limit=2)
        if len(sk) >= 2:
            a, b = sorted(sk[:2])
            keys.append(f"sk_{a}_{b}")
        elif len(sk) == 1:
            keys.append(f"sk_{sk[0]}")

    return keys


KEY_WEIGHTS = {
    "tok_": 2.0,
    "num_": 4.0,
    "zip_": 3.0,
    "bg_": 5.0,
    "tp_": 3.0,
    "sk_": 4.0,
}


def key_hash(country: str, key: str) -> int:
    """
    Deterministic 64-bit signed hash of (country, key) - used as cand_keys'
    storage key instead of the raw "key TEXT, country TEXT" pair, which cost
    ~2x the space of the integers it hashes to (measured: 5.57 GB for 57.3M
    rows in the old TEXT format). Must be a *process-independent* hash
    (unlike Python's hash()/dict-iteration order, which is randomized per
    process) since the index-build process and every later query process
    have to agree on it byte-for-byte.
    """
    digest = hashlib.blake2b(f"{country}|{key}".encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "big")
    return val - (1 << 64) if val >= (1 << 63) else val


class DiskBlockingEngine:
    def __init__(self, db_path: str = "output/candidates_index.db", max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.db_path = db_path
        self.max_candidates = max_candidates
        self.conn = None

    def build_index(self, s2_path: str, s3_path: str, reset: bool = True,
                     translit_dict: Dict[str, str] = None):
        """Ingest S2 and S3 candidate files and build B-Tree indexes on disk.

        translit_dict: learned Indic-script -> Latin word map (see
        transliterate.py), applied to each candidate's raw name/address
        before normalization - see DuckDBBlockingEngine.build_index()'s
        docstring for the full rationale (both engines share it so they
        can't silently diverge). Defaults to loading
        data_split/translit_dict.json if present.
        """
        if translit_dict is None:
            translit_dict = load_translit_dict()
        if reset and os.path.exists(self.db_path):
            os.remove(self.db_path)

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        c = self.conn.cursor()

        # Performance pragmas for bulk ingestion. journal_mode=OFF (no
        # rollback journal at all, not even in memory) is safe here because
        # this whole method only ever runs as a from-scratch rebuild - a
        # failed run just gets `reset=True`'d and redone, there's no partial
        # state worth protecting. This matters because the earlier
        # journal_mode=MEMORY choice buffered the multi-hundred-MB rollback
        # journal for the key-frequency-capping DELETE below in RAM, which
        # pushed a 7.4 GB host into full swap during that step.
        c.execute("PRAGMA synchronous = OFF;")
        c.execute("PRAGMA journal_mode = OFF;")
        c.execute("PRAGMA cache_size = 100000;")  # ~100MB cache

        print(f"Creating tables in {self.db_path}...")
        c.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                rid INTEGER PRIMARY KEY,
                cid TEXT UNIQUE,
                name TEXT,
                addr TEXT,
                country TEXT,
                norm_name TEXT,
                norm_addr TEXT,
                comp_name TEXT
            );
        """)
        # Staging table for blocking-key hashes before the frequency cap is
        # applied - plain heap rows (no PK yet) for fast bulk insert. `h`
        # already encodes the country (key_hash() hashes "country|key"), so
        # grouping by `h` alone later correctly caps postings per
        # (country, key), the same as the old per-(country, key) capping.
        c.execute("CREATE TABLE IF NOT EXISTS cand_keys_raw (h INTEGER, rid INTEGER);")

        # Pass 1: ingest raw records, normalize, and assign each one a
        # sequential integer rid - cheaper to store/join in cand_keys than
        # the TEXT cid (see key_hash()'s docstring for the storage-size
        # motivation). No blocking keys are generated yet: the rarity-based
        # address-token selection in build_blocking_keys() needs each
        # token's document frequency across the *whole* candidate pool,
        # which isn't known until every record has been normalized.
        next_rid = itertools.count()
        for path in [s2_path, s3_path]:
            if not os.path.exists(path):
                continue
            print(f"Ingesting candidates from: {path}...")
            t0 = time.time()
            cand_rows = []
            count = 0

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
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
                    # Computed and persisted once here (not just used
                    # locally) so retrieve_candidates()/get_by_id() can hand
                    # it back to callers instead of every feature-extraction
                    # call re-normalizing the same candidate's address again.
                    norm_addr = normalize_address(translit_addr)

                    cand_rows.append((next(next_rid), cid, raw_name, raw_addr, country, norm_name, norm_addr, comp_name))

                    count += 1
                    if len(cand_rows) >= 50000:
                        c.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", cand_rows)
                        self.conn.commit()
                        cand_rows = []
                        print(f"  Ingested {count} records... ({time.time()-t0:.1f}s)", end="\r")

                if cand_rows:
                    c.executemany("INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", cand_rows)
                    self.conn.commit()

            print(f"\nFinished ingesting {count} records from {os.path.basename(path)} in {time.time()-t0:.1f}s.")

        # Indexes on `candidates` are independent of cand_keys and cheap
        # (10.3M rows), so build those first. `cid TEXT UNIQUE` above
        # already gives get_by_id()/retrieve_candidates()'s exact-cid
        # lookups an index for free.
        print("Building B-Tree indexes on candidates table...")
        t0 = time.time()
        c.execute("CREATE INDEX IF NOT EXISTS idx_cand_norm ON candidates (country, norm_name);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cand_comp ON candidates (country, comp_name);")
        self.conn.commit()
        print(f"Candidates indexes built in {time.time()-t0:.1f}s.")

        # Pass 2: address-token document frequency per country, computed in
        # one read-only scan over the now-normalized candidates table.
        # build_blocking_keys() needs this to pick the *rarest* address
        # tokens as key anchors instead of just the positionally-first
        # ones (see _rare_addr_tokens - this is what makes an address's
        # chosen anchor tokens order-invariant AND consistent regardless of
        # which record happens to be indexed vs. queried). Computed in
        # Python (not SQL) since SQLite has no built-in word tokenizer;
        # persisted to addr_df so retrieve_candidates(), running in a
        # different process, reads the exact same rarity ranking instead of
        # a per-query guess that could silently diverge from build time.
        print("Computing address-token document frequency...")
        t0 = time.time()
        df_counter: Dict[Tuple[str, str], int] = {}
        for country, norm_addr in c.execute("SELECT country, norm_addr FROM candidates"):
            for w in _address_token_candidates(norm_addr):
                key = (country, w)
                df_counter[key] = df_counter.get(key, 0) + 1
        c.execute("""
            CREATE TABLE addr_df (
                country TEXT, token TEXT, df INTEGER,
                PRIMARY KEY (country, token)
            ) WITHOUT ROWID;
        """)
        c.executemany(
            "INSERT INTO addr_df VALUES (?, ?, ?)",
            ((country, token, df) for (country, token), df in df_counter.items()),
        )
        self.conn.commit()
        print(f"addr_df built in {time.time()-t0:.1f}s ({len(df_counter):,} (country, token) rows).")

        # Pass 3: generate blocking keys for every candidate now that
        # df_counter is available, and insert their hashes into the
        # staging table. Reads are done in bounded rid-ordered chunks,
        # fully fetched (.fetchall()) before any write for that chunk -
        # interleaving a live, not-yet-exhausted SELECT statement with
        # writes on the same connection is unsafe in SQLite, so each
        # chunk's read is completed first, then its keys are written.
        print("Generating blocking keys...")
        t0 = time.time()
        chunk_size = 200000
        last_rid = -1
        gen_count = 0
        while True:
            rows = c.execute(
                "SELECT rid, country, norm_name, norm_addr FROM candidates "
                "WHERE rid > ? ORDER BY rid LIMIT ?",
                (last_rid, chunk_size),
            ).fetchall()
            if not rows:
                break
            key_rows = []
            for rid, country, norm_name, norm_addr in rows:
                for key in build_blocking_keys(norm_name, norm_addr, country, df_counter):
                    key_rows.append((key_hash(country, key), rid))
            c.executemany("INSERT INTO cand_keys_raw VALUES (?, ?)", key_rows)
            self.conn.commit()
            last_rid = rows[-1][0]
            gen_count += len(rows)
            print(f"  Keyed {gen_count} records... ({time.time()-t0:.1f}s)", end="\r")
        print(f"\nBlocking keys generated in {time.time()-t0:.1f}s.")

        # Cap keys with more than MAX_TOKEN_POSTINGS postings (near-stopword
        # tokens) by filtering into a fresh table rather than DELETE-ing in
        # place.
        #
        # An earlier version of this built the cand_keys index first, on
        # the theory that it would speed up the GROUP BY below. It did -
        # but it also meant every one of the (potentially tens of millions
        # of) rows this step removes now had to pay B-Tree rebalancing cost
        # too, which measured far worse in practice: 2961s at full test
        # scale, vs 346.6s when there was no index yet to maintain.
        #
        # The first attempt at fixing that used a single
        # `WHERE (country, key) NOT IN (SELECT ... GROUP BY ... HAVING ...)`
        # query - which measured *far* worse still (SQLite doesn't reliably
        # materialize a GROUP-BY-aggregated multi-column NOT-IN subquery
        # into a one-time hash/index; it can re-run the whole aggregate
        # per outer row, which is why this hung for 90+ minutes with no
        # end in sight instead of the expected tens of seconds). The fix:
        # materialize the (small) list of over-frequent hashes into its own
        # indexed temp table *first*, then use a LEFT JOIN ... IS NULL
        # anti-join against it - a pattern SQLite's planner reliably turns
        # into one GROUP BY pass plus one indexed-lookup join, not O(n^2).
        print(f"Capping keys with more than {MAX_TOKEN_POSTINGS} postings...")
        t0 = time.time()
        c.execute("DROP TABLE IF EXISTS bad_hashes;")
        c.execute("DROP TABLE IF EXISTS cand_keys;")
        c.execute(
            "CREATE TEMP TABLE bad_hashes AS SELECT h FROM cand_keys_raw GROUP BY h HAVING COUNT(*) > ?;",
            (MAX_TOKEN_POSTINGS,),
        )
        c.execute("CREATE INDEX idx_bad_hashes ON bad_hashes (h);")
        # cand_keys is WITHOUT ROWID, clustered by (h, rid): no separate
        # index needed for the k.h IN (...) lookups retrieve_candidates()
        # does, and no per-row rowid to store on top of the two integers.
        c.execute("CREATE TABLE cand_keys (h INTEGER, rid INTEGER, PRIMARY KEY (h, rid)) WITHOUT ROWID;")
        c.execute(
            """
            INSERT OR IGNORE INTO cand_keys (h, rid)
            SELECT ck.h, ck.rid FROM cand_keys_raw ck
            LEFT JOIN bad_hashes bh ON ck.h = bh.h
            WHERE bh.h IS NULL
            ORDER BY ck.h, ck.rid;
            """
        )
        c.execute("DROP TABLE cand_keys_raw;")
        c.execute("DROP TABLE bad_hashes;")
        self.conn.commit()
        print(f"Capped in {time.time()-t0:.1f}s.")

        # VACUUM reclaims the space freed by the DROP TABLE/rename dance in
        # the capping step above (dropping cand_keys_raw/bad_hashes leaves
        # the old pages in the file's freelist rather than shrinking it).
        # Measured effect on a full-scale rebuild of the old TEXT-keyed
        # schema: 6.08 GB -> 4.84 GB for the exact same data, for free.
        print("Vacuuming database to reclaim freed space...")
        t0 = time.time()
        self.conn.execute("VACUUM;")
        print(f"Vacuumed in {time.time()-t0:.1f}s.")

        # Guardrail: log final size/row-counts and warn loudly if the index
        # is large relative to available RAM. A DB that doesn't fit in page
        # cache thrashes on every query - the run that prompted this check
        # measured 31-34 ent/s (a ~14-15hr full run) instead of the
        # ~200-300 ent/s baseline, and that wasn't visible until 90 minutes
        # in. This catches it in seconds instead.
        cand_count = c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        key_count = c.execute("SELECT COUNT(*) FROM cand_keys").fetchone()[0]
        db_size = os.path.getsize(self.db_path)
        print(
            f"Index built: {cand_count:,} candidates, {key_count:,} keys "
            f"({key_count / max(cand_count, 1):.2f} keys/candidate), "
            f"{db_size / 1e9:.2f} GB on disk."
        )
        try:
            with open("/proc/meminfo") as f:
                total_ram = int(f.readline().split()[1]) * 1024
            if db_size > 0.6 * total_ram:
                print(
                    f"WARNING: index size ({db_size / 1e9:.2f} GB) exceeds 60% of "
                    f"detected total RAM ({total_ram / 1e9:.2f} GB). Queries will "
                    f"likely thrash the page cache and be far slower than expected. "
                    f"pipeline.py's retrieval throughput smoke-test (run right before "
                    f"the full streaming loop) will catch this concretely - do not "
                    f"proceed to a full run if it reports low ent/s."
                )
        except (OSError, ValueError, IndexError):
            pass  # /proc/meminfo not available (non-Linux) - skip the check

    def open(self):
        """Open read-only connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA query_only = ON;")
            self.conn.execute("PRAGMA cache_size = 50000;")

    def get_by_id(self, cid: str):
        """Look up a single candidate record by id.

        Returns (name, addr, country, norm_name, norm_addr) or None. The
        norm_name/norm_addr are the already-normalized strings computed once
        at index-build time - pass them to extract_pair_features()'s
        cand_norm_name/cand_norm_addr instead of re-normalizing from raw.
        """
        if self.conn is None:
            self.open()
        row = self.conn.execute(
            "SELECT name, addr, country, norm_name, norm_addr FROM candidates WHERE cid = ?", (cid,)
        ).fetchone()
        return row

    def retrieve_candidates(self, s1_rec: Dict[str, str]) -> List[Tuple[str, str, str, str, str]]:
        """Retrieve candidate tuples (cid, name, addr, norm_name, norm_addr) for an S1 entity.

        norm_name/norm_addr are the candidate's already-normalized strings,
        computed once at index-build time and persisted in the candidates
        table - returned here so callers can pass them straight into
        extract_pair_features()'s cand_norm_name/cand_norm_addr instead of
        re-normalizing the same candidate's raw text on every pair (an S1
        with ~30 candidates was re-normalizing 30x redundantly before).
        """
        if self.conn is None:
            self.open()

        country = s1_rec["country"]
        raw_name = s1_rec["business_name"]
        raw_addr = s1_rec["business_address"]

        norm_name = normalize_business_name(raw_name)
        comp_name = "".join(norm_name.split()) if len(norm_name.split()) > 1 else ""

        c = self.conn.cursor()
        cand_map = {}  # cid -> (name, addr, norm_name, norm_addr, score)

        # 1. Exact Name match (Priority 1)
        if norm_name:
            res = c.execute(
                "SELECT cid, name, addr, norm_name, norm_addr FROM candidates "
                "WHERE country = ? AND norm_name = ? ORDER BY cid LIMIT 30",
                (country, norm_name)
            ).fetchall()
            for cid, name, addr, c_norm_name, c_norm_addr in res:
                cand_map[cid] = (name, addr, c_norm_name, c_norm_addr, 20.0)

        # 2. Compressed Name match
        if comp_name:
            res = c.execute(
                "SELECT cid, name, addr, norm_name, norm_addr FROM candidates "
                "WHERE country = ? AND comp_name = ? ORDER BY cid LIMIT 30",
                (country, comp_name)
            ).fetchall()
            for cid, name, addr, c_norm_name, c_norm_addr in res:
                prev_score = cand_map[cid][4] if cid in cand_map else 0.0
                cand_map[cid] = (name, addr, c_norm_name, c_norm_addr, prev_score + 15.0)

        # 3. Keys query (tokens, address numbers, zip codes, bigrams, name-
        #    token pairs, cross-script skeletons) - same key generator used
        #    at index-build time, so what a query produces is guaranteed to
        #    match what a record was indexed under. All keys for this
        #    record are looked up in a single round-trip (`h IN (...)`)
        #    instead of one query per key - at full test scale (1.7M S1
        #    entities) one query per key measured ~33ms/record (would be
        #    15+ hours end to end); batched, it's back down to the low
        #    single-digit ms/record the rest of the pipeline was designed
        #    around.
        norm_addr = normalize_address(raw_addr)

        # Address-token rarity must use the *same* document-frequency
        # ranking build_index() used, or a record could be indexed under
        # one rare-token anchor and queried under a different one. Only
        # this record's own candidate tokens need fetching (a handful of
        # words), not the whole addr_df table.
        addr_word_candidates = _address_token_candidates(norm_addr)
        df_lookup: Dict[Tuple[str, str], int] = {}
        if addr_word_candidates:
            placeholders = ",".join("?" for _ in addr_word_candidates)
            for token, df in c.execute(
                f"SELECT token, df FROM addr_df WHERE country = ? AND token IN ({placeholders})",
                [country, *addr_word_candidates],
            ).fetchall():
                df_lookup[(country, token)] = df

        query_keys = build_blocking_keys(norm_name, norm_addr, country, df_lookup)
        if query_keys:
            # Map each query key's hash back to its prefix (for weighting)
            # in Python - cand_keys itself no longer stores the key string,
            # only its hash, so this is the only place that association is
            # still needed.
            hash_to_prefix = {
                key_hash(country, key): key.split("_", 1)[0] + "_"
                for key in query_keys
            }
            placeholders = ",".join("?" for _ in hash_to_prefix)
            res = c.execute(
                f"SELECT k.h, c.cid, c.name, c.addr, c.norm_name, c.norm_addr "
                f"FROM cand_keys k JOIN candidates c ON k.rid = c.rid "
                f"WHERE k.h IN ({placeholders})",
                list(hash_to_prefix.keys()),
            ).fetchall()
            for h, cid, name, addr, c_norm_name, c_norm_addr in res:
                weight = KEY_WEIGHTS.get(hash_to_prefix.get(h, ""), 2.0)
                prev_score = cand_map[cid][4] if cid in cand_map else 0.0
                cand_map[cid] = (name, addr, c_norm_name, c_norm_addr, prev_score + weight)

        if not cand_map:
            return []

        # Rerank by text similarity before truncating, instead of truncating
        # on the raw summed key-weight score. The key-weight score is a
        # coarse proxy (how many/which keys matched) that frequently ranked
        # a same-key-but-dissimilar candidate above a true match sharing
        # fewer keys - measured on a 2,000-entity validation sample, 8.0% of
        # all true pairs were retrieved into the pool but then cut before
        # the top MAX_CANDIDATES_PER_S1. Only the top RERANK_POOL_SIZE by
        # key-weight are rescored (rapidfuzz cost is O(pool), and the pool
        # is already sorted by key-weight so this keeps the highest-signal
        # candidates), and this always runs on a pool built by an exact
        # index lookup - it can only reorder/select, never introduce a
        # candidate the key lookup didn't already find.
        s1_nums = extract_numbers(norm_addr)
        by_key_weight = sorted(cand_map.items(), key=lambda x: x[1][4], reverse=True)
        pool = by_key_weight[:RERANK_POOL_SIZE]

        def _sim_score(item):
            cid, (name, addr, c_norm_name, c_norm_addr, key_weight) = item
            name_sim = fuzz.token_set_ratio(norm_name, c_norm_name)
            addr_sim = fuzz.token_set_ratio(norm_addr, c_norm_addr)
            num_bonus = 10.0 if (s1_nums & extract_numbers(c_norm_addr)) else 0.0
            # Small key-weight tiebreak so equally-similar candidates keep
            # the original ranking's preference for more/stronger key hits.
            return 0.55 * name_sim + 0.35 * addr_sim + num_bonus + 0.01 * key_weight

        sorted_cands = sorted(pool, key=_sim_score, reverse=True)
        return [
            (cid, data[0], data[1], data[2], data[3])
            for cid, data in sorted_cands[:self.max_candidates]
        ]
