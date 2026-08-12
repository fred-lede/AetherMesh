from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("AIIH_DB_PATH", tempfile.mktemp(suffix=".db"))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
