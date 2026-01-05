# from __future__ import annotations

# # -----------------------------
# # Pass network viz defaults
# # -----------------------------

# # Pitch look
# DEFAULT_PITCH_TYPE: str = "wyscout"
# DEFAULT_PITCH_BG: str = "#1f2326"
# DEFAULT_PITCH_LINE_COLOR: str = "#4a4f52"

# # Colormap / color processing
# DEFAULT_CMAP_NAME: str = "RdYlGn"
# DEFAULT_COLOR_SOFTEN: float = 0.28   # mix toward pitch background (0..1)
# DEFAULT_COLOR_ALPHA: float = 0.78    # alpha applied to ramp colors (0..1)

# # Edge styling
# DEFAULT_EDGE_ALPHA_MIN: float = 0.45
# DEFAULT_EDGE_ALPHA_MAX: float = 0.92
# DEFAULT_EDGE_LW_MIN: float = 0.30
# DEFAULT_EDGE_LW_MAX: float = 2.60

# # Donut node styling
# DEFAULT_DONUT_RING_LW: float = 6
# DEFAULT_DONUT_HOLE_RATIO: float = 0.55

# # Labels
# DEFAULT_LABEL_FONTSIZE: int = 9
# DEFAULT_LABEL_OPACITY: float = 0.92
# DEFAULT_LABEL_BBOX: bool = False
# DEFAULT_LABEL_BBOX_ALPHA: float = 0.2
# DEFAULT_LABEL_BBOX_PAD: float = 0.16
# DEFAULT_LABEL_Y_OFFSET: float = 0.0

# # Node size scaling (power scale emphasis)
# DEFAULT_NODE_SIZE_MIN: float = 420.0
# DEFAULT_NODE_SIZE_MAX: float = 1850.0
# DEFAULT_NODE_SIZE_GAMMA: float = 0.70  # <1 emphasizes big nodes

from __future__ import annotations

DEFAULT_PITCH_TYPE: str = "wyscout"
DEFAULT_PITCH_BG: str = "#1f2326"
DEFAULT_PITCH_LINE_COLOR: str = "#4a4f52"

# Colormap / color processing
DEFAULT_CMAP_NAME: str = "RdYlGn"
DEFAULT_COLOR_SOFTEN: float = 0.14
DEFAULT_COLOR_ALPHA: float = 0.92

# Edge styling
DEFAULT_EDGE_ALPHA_MIN: float = 0.50
DEFAULT_EDGE_ALPHA_MAX: float = 0.95
DEFAULT_EDGE_LW_MIN: float = 0.55
DEFAULT_EDGE_LW_MAX: float = 3.40

# Donut node styling
DEFAULT_DONUT_RING_LW: float = 7.0
DEFAULT_DONUT_HOLE_RATIO: float = 0.50

# Labels
DEFAULT_LABEL_FONTSIZE: int = 10
DEFAULT_LABEL_OPACITY: float = 1.0
DEFAULT_LABEL_BBOX: bool = False
DEFAULT_LABEL_BBOX_ALPHA: float = 0.35
DEFAULT_LABEL_BBOX_PAD: float = 0.18
DEFAULT_LABEL_Y_OFFSET: float = 0.0


# Node size scaling (power scale emphasis)
DEFAULT_NODE_SIZE_MIN: float = 420.0
DEFAULT_NODE_SIZE_MAX: float = 1850.0
DEFAULT_NODE_SIZE_GAMMA: float = 0.70  # <1 emphasizes big nodes
