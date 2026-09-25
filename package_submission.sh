#!/usr/bin/env bash
# Package Submission Script for Amazon ML Challenge 2026

set -e

TEAM_NAME="cloudyrelic"
ZIP_NAME="${TEAM_NAME}_submission.zip"

echo "========================================================"
echo " Packaging Submission: $ZIP_NAME"
echo "========================================================"

# 1. Validate submission files
echo "Running validation checks..."
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test

# 2. Remove old zip if present
rm -f "$ZIP_NAME"

# 3. Create zip with exact structure required by competition:
# <team_name>_submission.zip
# ├── output/
# │   ├── matching_results.tsv
# │   └── candidate_pairs.tsv
# ├── code/
# │   └── business_entity_resolution/
# │       ├── src/
# │       ├── README.md
# │       └── requirements.txt
# └── Documentation_template.md

echo "Creating submission zip archive..."
zip -r "$ZIP_NAME" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/src/ \
    code/business_entity_resolution/README.md \
    code/business_entity_resolution/requirements.txt \
    Documentation_template.md

echo ""
echo "========================================================"
echo " Submission package created successfully: $ZIP_NAME"
echo " Size: $(du -h "$ZIP_NAME" | cut -f1)"
echo "========================================================"
