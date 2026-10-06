import os
import sys
from pathlib import Path

# Redirect to inner project if started from parent
INNER_DIR = Path(__file__).parent / "multimodal-deepfake-detector-master"
if INNER_DIR.exists():
    os.chdir(INNER_DIR)
    sys.path.insert(0, str(INNER_DIR))

# Execute the inner app.py
inner_app = INNER_DIR / "app.py"
if inner_app.exists():
    with open(inner_app, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
