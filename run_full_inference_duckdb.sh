#!/usr/bin/env bash
# Single-Process DuckDB Full Test Inference Runner for Amazon ML Challenge 2026.
#
# Drop-in replacement for run_full_inference.sh: same models, same cascade,
# same output/matching_results.tsv + output/candidate_pairs.tsv contract.
# Difference is the blocking engine only - DuckDB batched joins per chunk
# instead of SQLite per-entity queries (see src/duckdb_blocking.py).
#
# Deliberately single-process, unlike run_full_inference_fast.sh's
# multiprocessing version: measured on this box (7.4GB RAM, ~5.6GB index)
# that multiple worker processes each opening their own DuckDB connection
# fragmented the page cache and made things *worse* (~82 ent/s with 2
# workers vs ~255-270 ent/s single-process, confirmed on a real 30K-entity
# sample against the full index). One process gets the whole page cache to
# itself instead of splitting it.
#
# Reuses output/candidates_index.duckdb if it already exists (built once,
# took ~24 min at full scale) - pass --rebuild-index to force a rebuild.

set -e

cd "/home/cloudyrelic/amazon_ml"

source venv/bin/activate
export PYTHONPATH=code/business_entity_resolution
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo " Starting Single-Process DuckDB Full Test Inference Pipeline"
echo " Start Time: $(date)"
echo " Environment: $(which python3)"
echo "=========================================================="

# Same threshold as run_full_inference.sh - see that file's note on where
# 0.75 came from. Blocking engine changed, not the scoring cascade or its
# tuned threshold.
python3 -u -m src.pipeline_duckdb --threshold 0.75 "$@"

echo ""
echo "=========================================================="
echo " Inference Completed: $(date)"
echo " Ready to package and submit via: ./package_submission.sh"
echo "=========================================================="
