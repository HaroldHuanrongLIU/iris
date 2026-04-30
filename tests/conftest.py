from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)
