#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# VSLAM-LAB fork: SALAD and the MIT-SPARK fork of VGGT are pinned git submodules under third_party/
# (upstream clones their default branches here). Perception Encoder and SAM 3 are only needed for
# the optional open-set object detection (--run_os) and are not installed by this script.

# 1. Install Python dependencies
echo "Installing base requirements..."
pip3 install -r requirements.txt

# 2. Fetch the pinned third-party sources
echo "Fetching third-party submodules..."
git submodule update --init --recursive

# 3. Install Salad
echo "Installing Salad..."
pip install -e third_party/salad

# 4. Install our fork of VGGT
echo "Installing VGGT..."
pip install -e third_party/vggt

# 5. Install current repo in editable mode
echo "Installing current repo..."
pip install -e .

echo "Installation Complete"
