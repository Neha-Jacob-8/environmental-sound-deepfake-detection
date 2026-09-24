"""Paste-able Colab cells to run the FULL pipeline (fetch -> train all 4
models -> evaluate) on the full ~9,900-clip EnvSDD subset, using Colab's
fast network + free GPU instead of a slow home connection + CPU-only laptop.

Run each block as a separate cell, in order. Uses Google Drive to persist
data/checkpoints across sessions, since free Colab sessions can disconnect
(idle timeout / 12h cap) - if that happens, just reconnect, remount Drive,
re-run cells 1-3, and re-run the training cell with --resume (already
included below) to continue instead of starting over.
"""

# --- Cell 1: runtime + drive ---------------------------------------------
# Runtime menu -> Change runtime type -> T4 GPU, before running this.
CELL_1 = r'''
from google.colab import drive
drive.mount("/content/drive")

import torch
print("GPU available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

PROJECT_DIR = "/content/drive/MyDrive/envsdd_project/speech"
'''

# --- Cell 2: get the project onto Drive -----------------------------------
# Upload your project ZIP first (folder icon on the left -> upload to
# /content), then unzip it into Drive so it survives session restarts.
CELL_2 = r'''
import os
os.makedirs("/content/drive/MyDrive/envsdd_project", exist_ok=True)
if not os.path.isdir(PROJECT_DIR):
    !unzip -q "/content/environmental_sound_deepfake_part3_detection_models.zip" -d /content/drive/MyDrive/envsdd_project
%cd {PROJECT_DIR}
!pip install -q -r requirements.txt
'''

# --- Cell 3: fetch the full dataset (fast on Colab's network) ------------
CELL_3 = r'''
%cd {PROJECT_DIR}
!chmod +x run_preprocessing.sh
!./run_preprocessing.sh          # default sizes = full ~9,900 clips
# Re-run this cell again if it stops partway (network hiccup, session
# reconnect) - already-downloaded clips are skipped, it just fills gaps.
'''

# --- Cell 4: train all four models on GPU ---------------------------------
# Runs in the foreground so you can see progress; re-run any line with
# --resume added if a session disconnects mid-model.
CELL_4 = r'''
%cd {PROJECT_DIR}
!python3 -m src.training.smoke_test

!python3 -m src.training.train --model logmel_cnn          --epochs 30 --batch_size 32
!python3 -m src.training.train --model aasist       --epochs 30 --batch_size 32
!python3 -m src.training.train --model beats_aasist --epochs 15 --batch_size 32
!python3 -m src.training.train --model fusion       --epochs 15 --batch_size 32 \
    --cnn_checkpoint results/models/cnn_best.pt \
    --beats_aasist_checkpoint results/models/beats_aasist_best.pt

# If a cell above gets interrupted partway, re-run just that line with
# --resume, e.g.:
#   !python3 -m src.training.train --model aasist --epochs 30 --batch_size 32 --resume
'''

# --- Cell 5: evaluate + compare, then package results for download -------
CELL_5 = r'''
%cd {PROJECT_DIR}
!python3 -m src.evaluation.evaluate --compare "results/models/*_best.pt"

!zip -qr /content/envsdd_results.zip results data/metadata/manifest.csv
from google.colab import files
files.download("/content/envsdd_results.zip")
# Also already saved under Drive at {PROJECT_DIR}/results/ - no need to
# re-download if you'd rather just browse it in Drive.
'''
