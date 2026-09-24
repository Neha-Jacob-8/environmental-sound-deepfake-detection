#!/usr/bin/env bash
# Train and evaluate every detection model, in the order fusion needs
# (its two branches initialise from the cnn and beats_aasist checkpoints,
# so those must exist first).
#
#   ./run_training.sh
#   EPOCHS_AASIST=40 ./run_training.sh
#   MODELS="logmel_cnn aasist" ./run_training.sh     # a subset, in this order
#
# On Windows without Git Bash/WSL, run the python commands directly - see
# the "Detection models" section of README.md.
set -e
cd "$(dirname "$0")"

EPOCHS_LOGMEL_CNN="${EPOCHS_LOGMEL_CNN:-40}"
EPOCHS_CNN="${EPOCHS_CNN:-20}"
EPOCHS_AASIST="${EPOCHS_AASIST:-20}"
EPOCHS_BEATS="${EPOCHS_BEATS:-15}"
EPOCHS_FUSION="${EPOCHS_FUSION:-15}"
MODELS="${MODELS:-logmel_cnn cnn aasist beats_aasist fusion}"

mkdir -p logs results/models results/tables

echo ">>> smoke test (shape/crash check before committing to a real run)"
python3 -u -m src.training.smoke_test 2>&1 | tee logs/smoke_test.log

for model in $MODELS; do
  echo
  case "$model" in
    logmel_cnn)   epochs="$EPOCHS_LOGMEL_CNN" ;;
    cnn)          epochs="$EPOCHS_CNN" ;;
    aasist)       epochs="$EPOCHS_AASIST" ;;
    beats_aasist) epochs="$EPOCHS_BEATS" ;;
    fusion)       epochs="$EPOCHS_FUSION" ;;
    *)            echo "unknown model: $model" >&2; exit 1 ;;
  esac

  echo ">>> training ${model} (${epochs} epochs)"
  if [ "$model" = "fusion" ]; then
    # Seed both branches from the already-trained Level 1 and Level 3 runs.
    python3 -u -m src.training.train --model fusion --epochs "$epochs" \
        --cnn_checkpoint results/models/cnn_best.pt \
        --beats_aasist_checkpoint results/models/beats_aasist_best.pt \
        2>&1 | tee "logs/train_${model}.log"
  else
    python3 -u -m src.training.train --model "$model" --epochs "$epochs" \
        2>&1 | tee "logs/train_${model}.log"
  fi

  echo ">>> evaluating ${model} on the test split (seen G01-G04 vs unseen G05-G07)"
  python3 -u -m src.evaluation.evaluate \
      --checkpoint "results/models/${model}_best.pt" \
      2>&1 | tee "logs/evaluate_${model}.log"
done

echo
echo ">>> side-by-side comparison"
python3 -u -m src.evaluation.compare 2>&1 | tee logs/compare.log

echo
echo "Done. results/tables/model_comparison.csv has the summary."
