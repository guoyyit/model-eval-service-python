"""Application entry point: ``uvicorn main:app`` or ``fastapi dev``."""
from __future__ import annotations

import os

from app.api import create_app

app = create_app(
    db_path=os.environ.get("EVAL_DB_PATH", "eval.db"),
    max_workers=int(os.environ.get("EVAL_MAX_WORKERS", "4")),
    default_quota=int(os.environ.get("EVAL_DEFAULT_QUOTA", "100")),
    default_timeout=float(os.environ.get("EVAL_DEFAULT_TIMEOUT", "60")),
)
