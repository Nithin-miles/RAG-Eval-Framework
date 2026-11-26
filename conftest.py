"""pytest bootstrap: import path + a small temp-dir fixture.

We deliberately avoid pytest's built-in ``tmp_path`` here. On some Windows
setups (locked-down folders, antivirus interference) pytest's temporary
directory cleanup raises PermissionError and takes the whole test session down
with it. A plain folder under ``tests/_scratch`` is predictable everywhere and
is gitignored.
"""

import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRATCH = ROOT / "tests" / "_scratch"


@pytest.fixture
def run_dir() -> Path:
    """A fresh, writable directory for one test; cleaned up on a best-effort basis."""
    path = SCRATCH / uuid.uuid4().hex[:10]
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
