#!/bin/bash
# eval_with_splits.sh
# Usage: ./eval_with_splits.sh <model_name> <amber_dir>
# Example: ./eval_with_splits.sh molmo /your/AMBER

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <amber_dir>"
    echo "Example: $0 molmo /your/AMBER"
    exit 1
fi

MODEL=$1
AMBER_DIR=$2

cd "$AMBER_DIR" || { echo "❌ Cannot cd to $AMBER_DIR"; exit 1; }

# Default NLTK data path; override by setting NLTK_DATA before calling this script
if [ -z "$NLTK_DATA" ]; then
    export NLTK_DATA="$AMBER_DIR/nltk_data"
fi

echo "======================================"
echo "Model: ${MODEL}"
echo "AMBER dir: ${AMBER_DIR}"
echo "======================================"

for condition in original black_0.5 black_0.75 blur_0.5 blur_0.75 no_image; do
    echo ""
    echo "Processing: ${condition}"
    echo "--------------------------------------"

    python eval_amber_split.py \
        --amber_dir "$AMBER_DIR" \
        --input_file "outputs_generative_${MODEL}/outputs_${condition}.json" \
        --output_dir "evaluation_results_${MODEL}/split_analysis" \
        --model "${MODEL}" \
        --condition "${condition}" \
        --seed 42 \
        --n_splits 5 \
        --samples_per_split 200

    if [ $? -ne 0 ]; then
        echo "❌ Error processing ${condition}"
        exit 1
    fi
done

echo ""
echo "======================================"
echo "✅ All evaluations complete!"
echo "======================================"
echo "Results saved in: ${AMBER_DIR}/evaluation_results_${MODEL}/split_analysis/"
