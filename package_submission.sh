#!/usr/bin/env bash
# Package Submission Script for Amazon ML Challenge 2026

set -e

TEAM_NAME="Neural Collapse"
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
# │       ├── src/  (output-relevant modules only)
# │       ├── README.md
# │       └── requirements.txt
# ├── models/   (trained screener + specialist, 5MB; lets inference run without retraining)
# └── Documentation_template.md

echo "Creating submission zip archive..."
# Only src modules that directly contribute to the output: the DuckDB inference
# path plus the scripts that split the data and train the models it loads. The old SQLite
# pipeline, tuning, validation, splitting and test/eval scripts are left out.
SRC_DIR=code/business_entity_resolution/src
SRC_FILES=()
for m in __init__ config normalize transliterate blocking sqlite_blocking \
         duckdb_blocking features evaluate decode dedupe_matches \
         pipeline_duckdb split_data train_screener train_tf_specialist; do
    SRC_FILES+=("$SRC_DIR/$m.py")
done

FILES=(
    output/matching_results.tsv
    output/candidate_pairs.tsv
    "${SRC_FILES[@]}"
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    models/
    Documentation_template.md
)

if command -v zip >/dev/null 2>&1; then
    zip -r "$ZIP_NAME" "${FILES[@]}" -x "*/__pycache__/*"
else
    echo "'zip' not found, falling back to Python zipfile..."
    python3 - "$ZIP_NAME" "${FILES[@]}" <<'EOF'
import os, sys, zipfile
out, paths = sys.argv[1], sys.argv[2:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    z.write(os.path.join(root, f))
        else:
            z.write(p)
EOF
fi

echo ""
echo "========================================================"
echo " Submission package created successfully: $ZIP_NAME"
echo " Size: $(du -h "$ZIP_NAME" | cut -f1)"
echo "========================================================"
