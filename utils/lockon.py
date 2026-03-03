"""Lockon: AI-driven coarse selection zoomed on the character from a click point.

The user presses on the character; the lockon function uses an AI model (e.g. SAM)
to segment at the click and returns the coarse selection box so the view is zoomed on the character.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Any

# Type for any object that has fuzzy_select(frame, point) -> Optional[mask]
FuzzySelectSelector = Any


def compute_lockon_box(
    frame: np.ndarray,
    point: Tuple[int, int],
    selector: Optional[FuzzySelectSelector] = None,
    padding_ratio: float = 0.1,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute coarse selection box (x_min, y_min, x_max, y_max) from a click on the character.

    Uses an AI model (e.g. SAM via a selector with fuzzy_select) to segment at the click,
    then returns the bounding box of the segment so the view is zoomed on the character.

    Args:
        frame: Input frame as BGR numpy array (e.g. from video).
        point: (x, y) coordinate where the user clicked on the character.
        selector: Optional object with fuzzy_select(frame, point) returning a binary mask.
                  If None, returns None (caller can pass SAM selector from app).
        padding_ratio: Fraction of box size to add as padding (default 0.1).

    Returns:
        (x_min, y_min, x_max, y_max) bounding box within frame bounds containing the point,
        or None if lockon fails or selector is None.
    """
    if selector is None:
        return None
    if not hasattr(selector, "fuzzy_select") or not callable(getattr(selector, "fuzzy_select")):
        return None
    try:
        mask = selector.fuzzy_select(frame, point)
    except Exception:
        return None
    if mask is None or mask.size == 0:
        return None
    # Binary mask: foreground typically > 0
    binary = (mask > 128).astype(np.uint8)
    if np.sum(binary) == 0:
        return None
    height, width = frame.shape[:2]
    coords = np.argwhere(binary > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Add padding (zoom margin around character)
    w, h = x_max - x_min, y_max - y_min
    pad_x = max(1, int(w * padding_ratio))
    pad_y = max(1, int(h * padding_ratio))
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    # Ensure point is inside box (should already be)
    px, py = int(point[0]), int(point[1])
    x_min = min(x_min, px)
    y_min = min(y_min, py)
    x_max = max(x_max, px + 1)
    y_max = max(y_max, py + 1)
    x_min = max(0, min(x_min, width - 1))
    x_max = max(x_min + 1, min(x_max, width))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(y_min + 1, min(y_max, height))
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def is_valid_lockon_box(
    box: Optional[Tuple[int, int, int, int]],
    frame_shape: Tuple[int, ...],
    point: Tuple[int, int],
) -> bool:
    """Return True if box is a valid lockon coarse box for the given frame and click point.

    Valid means: 4-tuple, within frame bounds, and contains the click point.

    Args:
        box: (x_min, y_min, x_max, y_max) or None.
        frame_shape: (height, width) or (height, width, channels) from frame.shape.
        point: (x, y) click coordinate.

    Returns:
        True if box is valid.
    """
    if box is None:
        return False
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return False
    x_min, y_min, x_max, y_max = box
    try:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    except (TypeError, ValueError):
        return False
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x, y = int(point[0]), int(point[1])
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        return False
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        return False
    return True
