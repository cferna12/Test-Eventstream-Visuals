from __future__ import annotations

from typing import Literal

# -----------------------------
# Possession chain viz defaults
# -----------------------------
DEFAULT_PITCH_TYPE: str = "wyscout"

# Soccermatics-ish style controls (moved from endpoint params)
DEFAULT_PITCH_BG_COLOR: str = "#16313a"
DEFAULT_PITCH_LINE_COLOR: str = "#7f8c8d"

DEFAULT_BASE_EVENT_COLOR: str = "#cfd8dc"
DEFAULT_START_COLOR: str = "#4da3ff"
DEFAULT_END_COLOR: str = "#ff4d4d"
DEFAULT_SHOT_COLOR: str = "#ff4d4d"

DEFAULT_PASS_LW: float = 2.4
DEFAULT_CARRY_LW: float = 2.0

DEFAULT_NODE_SIZE: float = 175
DEFAULT_NODE_EDGE_LW: float = 2.0

DEFAULT_ALPHA_MIN: float = 0.30
DEFAULT_ALPHA_MAX: float = 0.95

DEFAULT_NUMBER_FONTSIZE: int = 9

# Shot arrow behavior defaults
DEFAULT_SHOT_TO_GOAL: bool = True
DEFAULT_GOAL_X: float = 100.0  # you noted goal always at x=100 in your coordinate system

# Approximate goal mouth y targets in 0..100 coordinates (tweak if you want)
DEFAULT_GOAL_Y_LEFT: float = 62.0
DEFAULT_GOAL_Y_CENTER: float = 50.0
DEFAULT_GOAL_Y_RIGHT: float = 38.0
