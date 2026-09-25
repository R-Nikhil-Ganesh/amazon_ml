#!/usr/bin/env bash
# Full Test Inference Runner for Amazon ML Challenge 2026

set -e

# Change to project root directory
cd "/home/cloudyrelic/amazon_ml"

# Activate Python environment
source venv/bin/activate
export PYTHONPATH=code/business_entity_resolution
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo " Starting Full Test Inference Pipeline"
echo " Start Time: $(date)"
echo " Environment: $(which python3)"
echo "=========================================================="

# Run cascaded inference across all 1,732,544 test entities
python3 -u -m src.pipeline --threshold 0.88

echo ""
echo "=========================================================="
echo " Inference Completed: $(date)"
echo " Ready to package and submit via: ./package_submission.sh"
echo "=========================================================="
