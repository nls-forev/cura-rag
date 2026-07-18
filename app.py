"""Hugging Face Space entry point.

Runs the whole system in one process: embedded Qdrant, a self-seeded corpus, and
a Gradio UI over the same Answerer the API uses. Set the sensible single-container
defaults before importing curarag so config picks them up. Override any of them
with real environment variables / Space secrets.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The package is not pip-installed on the Space (HF installs requirements before
# copying the repo), so make src/ importable directly.
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QDRANT_PATH", str(ROOT / "qdrant_data"))
os.environ.setdefault("SEED_ON_STARTUP", "true")
os.environ.setdefault("SEED_INCLUDE_OPENFDA", "false")
os.environ.setdefault("CURARAG_DATA_DIR", str(ROOT / "data"))

from curarag.gradio_ui import build_demo  # noqa: E402
from curarag.seed import ensure_seeded  # noqa: E402

ensure_seeded()
demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
