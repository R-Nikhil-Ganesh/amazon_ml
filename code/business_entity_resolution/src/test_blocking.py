#!/usr/bin/env python3
"""
Unit and Integration Tests for Blocking Engines.

Tests both:
1. BlockingEngine (in-memory candidate index for training/validation)
2. DiskBlockingEngine (disk-based SQLite candidate index for scalable test inference)
3. evaluate_blocking_recall calculation logic
"""

import csv
import os
import sys
import tempfile
import unittest

# Support running as a standalone script or module
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.blocking import BlockingEngine, evaluate_blocking_recall
    from src.sqlite_blocking import DiskBlockingEngine, build_blocking_keys, key_hash
    from src.normalize import normalize_business_name, normalize_address, extract_numbers, phonetic_skeleton
    from src.dedupe_matches import dedupe, resolve_winners
else:
    from .blocking import BlockingEngine, evaluate_blocking_recall
    from .sqlite_blocking import DiskBlockingEngine, build_blocking_keys, key_hash
    from .normalize import normalize_business_name, normalize_address, extract_numbers, phonetic_skeleton
    from .dedupe_matches import dedupe, resolve_winners


class TestNormalization(unittest.TestCase):
    """Test text and address normalization utilities used by blocking."""

    def test_business_name_normalization(self):
        self.assertEqual(normalize_business_name("Acme Corp. LLC"), "acme")
        self.assertEqual(normalize_business_name("  Google Inc. "), "google")
        self.assertEqual(normalize_business_name("Tata Motors Pvt Ltd"), "tata motors")

    def test_address_normalization(self):
        norm = normalize_address("123 N. Main St., Suite 400")
        self.assertIn("123", norm)
        self.assertIn("main", norm)

    def test_extract_numbers(self):
        nums = extract_numbers("100 Industrial Pkwy, Bldg 5, Ste 200")
        self.assertEqual(nums, {"100", "5", "200"})

    def test_accent_folding(self):
        # Accented and unaccented spellings of the same French word must
        # normalize identically, or blocking/features never see them as similar.
        self.assertEqual(
            normalize_business_name("Fédération de Velo SAS"),
            normalize_business_name("Federation de Velo SARL"),
        )
        self.assertEqual(
            normalize_business_name("OZT ÀMICALE SAS"),
            normalize_business_name("OZT Amicale S.A.S."),
        )

    def test_phonetic_skeleton_bridges_script(self):
        self.assertEqual(phonetic_skeleton("global"), phonetic_skeleton("ग्लोबल"))
        self.assertEqual(phonetic_skeleton("healthcare"), phonetic_skeleton("हेल्थकेयर"))

    def test_dotted_acronym_suffix_stripping(self):
        # clean_basic() turns "S.A.S." into "s a s" before suffix stripping;
        # that must still be recognized as the "sas" legal suffix.
        self.assertEqual(
            normalize_business_name("Engages Art Pharmacie S.C.I"),
            normalize_business_name("Engages Art Pharmacie SCI"),
        )


class TestBlockingKeyGenerator(unittest.TestCase):
    """Test the shared key generator used by both index-build and query time."""

    def test_reordered_address_same_keys(self):
        # "1064 Newton Rd, Iowa City, IA" vs "IA, Iowa City, 1064 Newton Rd" -
        # word order must not change whether these two addresses block together.
        n1 = normalize_business_name("Vision Partners Corp")
        a1 = normalize_address("IA, Iowa City, 1064 Newton Rd")
        a2 = normalize_address("1064 Newton Rd, Iowa City, IA")
        k1 = set(build_blocking_keys(n1, a1))
        k2 = set(build_blocking_keys(n1, a2))
        self.assertTrue(k1 & k2, "reordered addresses should share at least one blocking key")

    def test_number_keys_use_every_number_not_just_first(self):
        # Regression test for the `list(nums)[0]` bug: Python's set iteration
        # order is randomized per-process, so a key built off "the first"
        # number of a set could silently differ between index-build and
        # query time for the exact same address. Every number must produce a key.
        norm_addr = normalize_address("1424 Cottage View Ln, Unit 5")
        keys = build_blocking_keys("acme", norm_addr)
        self.assertTrue(any("1424" in k for k in keys))
        self.assertTrue(any("5" in k.split("_") for k in keys))

    def test_keys_are_deterministic_across_calls(self):
        norm_addr = normalize_address("1424 Cottage View Ln, Unit 5, Springfield")
        first = build_blocking_keys("acme corp", norm_addr)
        for _ in range(5):
            self.assertEqual(first, build_blocking_keys("acme corp", norm_addr))

    def test_rarity_ranking_is_order_invariant(self):
        # "commerce" is common (high df) across the pool, "zylotech" is rare.
        # Regardless of which one appears first in the address text, the
        # rare token should be the one chosen as the num_/bg_ anchor - a
        # positional (first-N) selection would instead pick whichever word
        # happens to come first, which differs between a record and its
        # reordered counterpart.
        df_lookup = {
            ("US", "commerce"): 5000, ("US", "way"): 4000,
            ("US", "park"): 3000, ("US", "dallas"): 2000,
            ("US", "zylotech"): 2,
        }
        addr1 = normalize_address("500 Commerce Way, Zylotech Park, Dallas")
        addr2 = normalize_address("Zylotech Park, Dallas, 500 Commerce Way")
        k1 = build_blocking_keys("acme", addr1, "US", df_lookup)
        k2 = build_blocking_keys("acme", addr2, "US", df_lookup)
        self.assertEqual(set(k1), set(k2))
        self.assertTrue(any("zylotech" in k for k in k1))

    def test_name_token_pair_key_recovers_capped_single_tokens(self):
        # tp_ pair keys exist independently of tok_ single-token keys, so a
        # match survives even if both individual tokens got capped as
        # near-stopword-frequent in a large pool.
        keys = build_blocking_keys("collins cornerstone services", "1 main street")
        self.assertIn("tp_collins_cornerstone", keys)

    def test_skeleton_key_bridges_script_mismatch(self):
        # India-only cross-script anchor: same business name in Latin and
        # Devanagari script should produce the same sk_ key.
        latin = build_blocking_keys("global developers", "1 main street", "India", {})
        deva = build_blocking_keys("ग्लोबल डेवलपर्स", "1 main street", "India", {})
        latin_sk = {k for k in latin if k.startswith("sk_")}
        deva_sk = {k for k in deva if k.startswith("sk_")}
        self.assertTrue(latin_sk)
        self.assertEqual(latin_sk, deva_sk)
        # Not generated outside SKELETON_KEY_COUNTRIES (US/France etc.).
        us_keys = build_blocking_keys("global developers", "1 main street", "US", {})
        self.assertFalse(any(k.startswith("sk_") for k in us_keys))

    def test_key_hash_deterministic_and_country_scoped(self):
        # key_hash must not depend on Python's per-process hash
        # randomization (unlike hash()/set iteration order) since the
        # index-build process and every later query process must agree on
        # it byte-for-byte.
        h1 = key_hash("US", "tok_acme")
        for _ in range(5):
            self.assertEqual(h1, key_hash("US", "tok_acme"))
        self.assertNotEqual(h1, key_hash("India", "tok_acme"))


class TestInMemoryBlockingEngine(unittest.TestCase):
    """Test BlockingEngine indexing and multi-pass candidate retrieval."""

    def setUp(self):
        self.engine = BlockingEngine(max_candidates=10)
        self.candidates = [
            {
                "entity_id": "S2-001",
                "business_name": "Acme Industrial Tools Inc",
                "business_address": "500 Commerce Way, Dallas, TX 75201",
                "country": "US",
            },
            {
                "entity_id": "S2-002",
                "business_name": "Davis Decker Construction",
                "business_address": "1200 Highland Ave, Austin, TX",
                "country": "US",
            },
            {
                "entity_id": "S3-003",
                "business_name": "Shiva Exports Pvt Ltd",
                "business_address": "Plot 42, Safal Solitaire, Ahmedabad, Gujarat",
                "country": "India",
            },
            {
                "entity_id": "S3-004",
                "business_name": "Unrelated Business",
                "business_address": "999 Faraway Blvd, Seattle, WA",
                "country": "US",
            },
        ]
        self.engine.index_candidates(self.candidates)

    def test_exact_name_match(self):
        query = {
            "entity_id": "S1-100",
            "business_name": "Acme Industrial Tools",
            "business_address": "500 Commerce Way, Dallas, Texas",
            "country": "US",
        }
        cand_ids = self.engine.retrieve_candidates_for_s1(query)
        self.assertIn("S2-001", cand_ids)

    def test_compressed_name_match(self):
        query = {
            "entity_id": "S1-101",
            "business_name": "DavisDecker Construction",
            "business_address": "Highland Ave, Austin",
            "country": "US",
        }
        cand_ids = self.engine.retrieve_candidates_for_s1(query)
        self.assertIn("S2-002", cand_ids)

    def test_address_bigram_anchor_match(self):
        # Even if name is in transliteration or abbreviated, address bigram captures it
        query = {
            "entity_id": "S1-102",
            "business_name": "Shiva Traders",
            "business_address": "42 Safal Solitaire, Ahmedabad",
            "country": "India",
        }
        cand_ids = self.engine.retrieve_candidates_for_s1(query)
        self.assertIn("S3-003", cand_ids)

    def test_country_partition_isolation(self):
        # US query should never retrieve an Indian candidate
        query = {
            "entity_id": "S1-103",
            "business_name": "Shiva Exports",
            "business_address": "Plot 42, Safal Solitaire",
            "country": "US",
        }
        cand_ids = self.engine.retrieve_candidates_for_s1(query)
        self.assertNotIn("S3-003", cand_ids)

    def test_max_candidates_limit(self):
        small_engine = BlockingEngine(max_candidates=2)
        small_engine.index_candidates(self.candidates)
        query = {
            "entity_id": "S1-104",
            "business_name": "Acme Davis Construction",
            "business_address": "Commerce Way, Highland Ave",
            "country": "US",
        }
        cand_ids = small_engine.retrieve_candidates_for_s1(query)
        self.assertLessEqual(len(cand_ids), 2)


class TestDiskBlockingEngine(unittest.TestCase):
    """Test DiskBlockingEngine (SQLite on disk) indexing and retrieval."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_candidates.db")
        self.s2_path = os.path.join(self.temp_dir.name, "s2.csv")
        self.s3_path = os.path.join(self.temp_dir.name, "s3.csv")

        # Write dummy CSV files
        with open(self.s2_path, "w", encoding="utf-8") as f:
            f.write("record_id,entity_id,name,address,country\n")
            f.write("1,S2-001,Acme Industrial Tools,500 Commerce Way,US\n")
            f.write("2,S2-002,Davis Decker Construction,1200 Highland Ave,US\n")

        with open(self.s3_path, "w", encoding="utf-8") as f:
            f.write("record_id,entity_id,name,address,country\n")
            f.write("1,S3-003,Shiva Exports,Plot 42 Safal Solitaire,India\n")

        self.engine = DiskBlockingEngine(db_path=self.db_path, max_candidates=10)
        self.engine.build_index(self.s2_path, self.s3_path, reset=True)

    def tearDown(self):
        if self.engine.conn is not None:
            self.engine.conn.close()
        self.temp_dir.cleanup()

    def test_retrieve_candidates(self):
        query = {
            "entity_id": "S1-100",
            "business_name": "Acme Industrial Tools",
            "business_address": "500 Commerce Way",
            "country": "US",
        }
        results = self.engine.retrieve_candidates(query)
        self.assertTrue(len(results) > 0)
        cid, name, addr, norm_name, norm_addr = results[0]
        self.assertEqual(cid, "S2-001")
        self.assertIn("Acme", name)
        self.assertIn("acme", norm_name)  # already-normalized, persisted at index-build time
        self.assertTrue(norm_addr)

    def test_cross_country_filtering(self):
        query = {
            "entity_id": "S1-101",
            "business_name": "Shiva Exports",
            "business_address": "Plot 42 Safal Solitaire",
            "country": "US",
        }
        results = self.engine.retrieve_candidates(query)
        retrieved_ids = [cid for cid, _, _, _, _ in results]
        self.assertNotIn("S3-003", retrieved_ids)

    def test_addr_df_table_built_and_used_for_rarity(self):
        # "commerce"/"highland" recur across the tiny fixture pool while a
        # one-off token doesn't - addr_df should reflect that, and it must
        # exist post-build for retrieve_candidates()'s query-time lookup.
        row = self.engine.conn.execute(
            "SELECT df FROM addr_df WHERE country = 'US' AND token = 'commerce'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row[0], 1)

    def test_cross_script_retrieval_end_to_end(self):
        # A Devanagari-spelled candidate must be retrievable by its Latin-
        # spelled S1 counterpart via the sk_ skeleton key, through the real
        # build_index() -> retrieve_candidates() path (not just the key
        # generator in isolation).
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "india.db")
            s2_path = os.path.join(tmp, "s2.csv")
            s3_path = os.path.join(tmp, "s3.csv")
            with open(s2_path, "w", encoding="utf-8") as f:
                f.write("record_id,entity_id,name,address,country\n")
                f.write("1,S2-501,ग्लोबल डेवलपर्स प्राइवेट लिमिटेड,Mulund East Mumbai,India\n")
            with open(s3_path, "w", encoding="utf-8") as f:
                f.write("record_id,entity_id,name,address,country\n")

            engine = DiskBlockingEngine(db_path=db_path, max_candidates=10)
            engine.build_index(s2_path, s3_path, reset=True)
            try:
                query = {
                    "entity_id": "S1-900",
                    "business_name": "Global Developers Private Limited",
                    "business_address": "Mulund East Mumbai",
                    "country": "India",
                }
                results = engine.retrieve_candidates(query)
                self.assertIn("S2-501", [cid for cid, _, _, _, _ in results])
            finally:
                engine.conn.close()


class TestEvaluateBlockingRecall(unittest.TestCase):
    """Test blocking recall calculation."""

    def test_recall_metric(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", delete=False) as f:
            f.write("record_id,source1_entity_id,matched_entity_ids\n")
            f.write('1,S1-1,"S2-1,S3-1"\n')
            f.write("2,S1-2,S2-2\n")
            gt_path = f.name

        try:
            candidates_map = {
                "S1-1": ["S2-1", "S2-99"],  # captured S2-1 (1 of 2)
                "S1-2": ["S2-2", "S2-50"],  # captured S2-2 (1 of 1)
            }
            recall, avg_cand, total_s1 = evaluate_blocking_recall(candidates_map, gt_path)
            self.assertEqual(total_s1, 2)
            self.assertAlmostEqual(recall, 2 / 3, places=2)
            self.assertEqual(avg_cand, 2.0)
        finally:
            if os.path.exists(gt_path):
                os.remove(gt_path)


class TestDedupeMatches(unittest.TestCase):
    """Test Stage 4 global one-S1-per-candidate resolution."""

    def _write(self, path, rows, header):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    def test_duplicate_candidate_kept_by_highest_scorer_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            matching_in = os.path.join(tmp, "matching_results.tsv")
            scores = os.path.join(tmp, "match_scores.tsv")
            matching_out = os.path.join(tmp, "matching_results_dedup.tsv")

            # S1-1 and S1-2 both claimed S2-1; S1-2 scored it higher.
            self._write(
                matching_in,
                [["S1-1", "S2-1,S3-1"], ["S1-2", "S2-1"]],
                ["source1_entity_id", "matched_entity_ids"],
            )
            self._write(
                scores,
                [["S1-1", "S2-1", "0.80"], ["S1-1", "S3-1", "0.95"], ["S1-2", "S2-1", "0.92"]],
                ["source1_entity_id", "cand_entity_id", "prob"],
            )

            before, after = dedupe(matching_in, scores, matching_out)
            self.assertEqual(before, 3)
            self.assertEqual(after, 2)

            with open(matching_out, "r", encoding="utf-8") as f:
                rows = {r[0]: r[1] for r in csv.reader(f, delimiter="\t")}
            self.assertEqual(rows["S1-1"], "S3-1")   # lost S2-1 to S1-2
            self.assertEqual(rows["S1-2"], "S2-1")   # kept it - higher score

    def test_no_duplicates_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            matching_in = os.path.join(tmp, "matching_results.tsv")
            scores = os.path.join(tmp, "match_scores.tsv")
            matching_out = os.path.join(tmp, "matching_results_dedup.tsv")

            self._write(
                matching_in,
                [["S1-1", "S2-1"], ["S1-2", ""]],
                ["source1_entity_id", "matched_entity_ids"],
            )
            self._write(
                scores,
                [["S1-1", "S2-1", "0.90"]],
                ["source1_entity_id", "cand_entity_id", "prob"],
            )

            before, after = dedupe(matching_in, scores, matching_out)
            self.assertEqual(before, 1)
            self.assertEqual(after, 1)

    def test_resolve_winners_keeps_max_prob(self):
        with tempfile.TemporaryDirectory() as tmp:
            scores = os.path.join(tmp, "match_scores.tsv")
            self._write(
                scores,
                [["S1-1", "S2-1", "0.5"], ["S1-2", "S2-1", "0.7"], ["S1-3", "S2-1", "0.6"]],
                ["source1_entity_id", "cand_entity_id", "prob"],
            )
            winners = resolve_winners(scores)
            self.assertEqual(winners["S2-1"], ("S1-2", 0.7))


if __name__ == "__main__":
    unittest.main()
