#!/usr/bin/env bash
# Fetch and verify the EnvSDD working subset.
#
# Safe to re-run: clips already on disk are skipped, so if the network drops
# just run it again and it picks up where it stopped.
#
#   ./run_preprocessing.sh              # default subset (~9,900 clips, ~1.2 GB)
#   TRAIN=400 VAL=100 TEST=100 ./run_preprocessing.sh    # smaller pilot
#
# Sizes are in SOURCE GROUPS, not clips:
#   train/validation group = 5 clips (1 real + G01-G04)
#   test group             = 8 clips (1 real + G01-G07)

set -u  # not -e: a failed split should not stop the others

cd "$(dirname "$0")"

TRAIN=${TRAIN:-1200}
VAL=${VAL:-300}
TEST=${TEST:-300}
WORKERS=${WORKERS:-8}

mkdir -p logs

echo "=== EnvSDD preprocessing ==="
echo "train=${TRAIN} groups (~$((TRAIN * 5)) clips)"
echo "val=${VAL} groups (~$((VAL * 5)) clips)"
echo "test=${TEST} groups (~$((TEST * 8)) clips)"
echo "total ~$((TRAIN * 5 + VAL * 5 + TEST * 8)) clips, ~$(((TRAIN * 5 + VAL * 5 + TEST * 8) * 128 / 1024)) MB"
echo

run_split () {
  local split=$1 groups=$2 chunks=$3
  echo ">>> fetching ${split} (${groups} groups) - logging to logs/fetch_${split}.log"
  python3 -u -m src.preprocessing.fetch_subset \
      --split "${split}" --groups "${groups}" --chunks "${chunks}" \
      --workers "${WORKERS}" 2>&1 | tee "logs/fetch_${split}.log"
  echo
}

run_split train      "${TRAIN}" 20
run_split validation "${VAL}"   10
run_split test       "${TEST}"  10

echo ">>> verifying"
python3 -u -m src.preprocessing.verify_subset 2>&1 | tee logs/verify.log

echo
echo ">>> standardisation report"
python3 -u -m src.preprocessing.audio_report 2>&1 | tee logs/audio_report.log

echo
echo ">>> waveform / log-Mel plots"
python3 -u -m src.preprocessing.visualize 2>&1 | tee logs/visualize.log

echo
echo "Done. If any split reports incomplete source groups or problem clips,"
echo "just run this script again - it resumes and fills the gaps."
