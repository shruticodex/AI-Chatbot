"""Start the FastAPI server."""

import os
import sys
from pathlib import Path
import uvicorn

# 1. Dynamically find the project root (one folder up from /scripts)
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Add the root to Python's system path
sys.path.insert(0, str(ROOT_DIR))
os.environ["PYTHONPATH"] = str(ROOT_DIR)

if __name__ == "__main__":
    # 3. Start the server and explicitly set app_dir
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(ROOT_DIR),
    )