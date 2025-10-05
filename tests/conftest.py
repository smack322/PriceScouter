import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
from pathlib import Path

def _load_env_once():
    # Load .env if present (local dev). In CI we rely on injected env vars.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)

_load_env_once()

os.environ.setdefault("DISABLE_LLM", "1")