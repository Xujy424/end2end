from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("Z:/") if Path("Z:/axis/dates.npy").is_file() else Path("/data/shanghai/xujiayi/workflow/data/")
