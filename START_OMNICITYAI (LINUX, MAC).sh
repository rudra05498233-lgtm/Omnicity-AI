#!/bin/bash

echo ""
echo "  ====================================================================="
echo "    OmniCity AI  |  Autonomous Urban Operating System  |  v4.2.0"
echo "  ====================================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  [!] ERROR: python3 not found."
    echo "      Install it from https://python.org or via: brew install python3"
    exit 1
fi
echo "  [OK] Python found: $(python3 --version)"
echo ""

echo "  [*] Installing required packages..."
pip3 install fastapi uvicorn sqlalchemy pillow opencv-python numpy \
    facenet-pytorch ultralytics transformers torch torchvision \
    scikit-learn pandas networkx easyocr httpx huggingface_hub \
    --quiet --disable-pip-version-check

echo "  [OK] Packages ready."
echo ""
echo "  [*] Starting OmniCity AI on http://localhost:8000 ..."
echo ""
echo "  ---------------------------------------------------------------"
echo "   Once running, open your browser:"
echo ""
echo "    http://localhost:8000/index.html         (Citizen Portal)"
echo "    http://localhost:8000/dashboard.html     (Operator Dashboard)"
echo "    http://localhost:8000/cctv_monitor.html  (CCTV Monitor)"
echo "    http://localhost:8000/citizen_walker.html (Walker View)"
echo ""
echo "   Press Ctrl+C to stop."
echo "  ---------------------------------------------------------------"
echo ""

cd "$(dirname "$0")/omni"
python3 backend.py
