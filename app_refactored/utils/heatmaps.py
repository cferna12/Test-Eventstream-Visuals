from __future__ import annotations

import os
from typing import Literal, List

# MUST match what you used in sql/01_bins_and_indexes.sql
BINS_X = int(os.environ.get("HEATMAP_BINS_X", "60"))
BINS_Y = int(os.environ.get("HEATMAP_BINS_Y", "40"))

ON_BALL_PRIMARY_DEFAULT: List[str] = [
    "touch", "pass", "received_pass", "duel", "interception", "shot", "carry",
    "clearance", "acceleration", "pressing_attempt", "infraction",
]

# -----------------------------
# Default rendering "knobs"
# (Keep frontend simple)
# -----------------------------
DEFAULT_PITCH_TYPE: str = "wyscout"

DEFAULT_BLUR: bool = True
DEFAULT_SIGMA: float = 1.5
DEFAULT_TRUNCATE: float = 4.0
DEFAULT_BLUR_MODE: Literal["nearest", "constant", "reflect", "mirror", "wrap"] = "nearest"

DEFAULT_CLIP_HI: float = 99.0
DEFAULT_SCALE: Literal["linear", "log", "power"] = "power"
DEFAULT_GAMMA: float = 1

DEFAULT_CMAP: str = "magma"
DEFAULT_ALPHA: float = 0.95
