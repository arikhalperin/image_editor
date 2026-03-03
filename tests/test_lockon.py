"""Tests for lockon: button, flow, and compute_lockon_box contract."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.lockon import compute_lockon_box, is_valid_lockon_box

# AppTest requires running from project root so app.py is found
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


# --- Contract: lockon module ---


def test_compute_lockon_box_exists_and_is_callable():
    """compute_lockon_box exists and is callable with (frame, point)."""
    assert callable(compute_lockon_box)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    point = (50, 50)
    result = compute_lockon_box(frame, point)
    # Stub returns None; when implemented, must be None or 4-tuple
    assert result is None or (
        isinstance(result, (tuple, list))
        and len(result) == 4
        and all(isinstance(v, (int, np.integer)) for v in result)
    )


def test_compute_lockon_box_return_type_contract():
    """Return type is Optional[Tuple[int, int, int, int]]."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    point = (10, 10)
    result = compute_lockon_box(frame, point)
    if result is not None:
        assert isinstance(result, (tuple, list)), "must be tuple or list"
        assert len(result) == 4, "must be 4-tuple (x_min, y_min, x_max, y_max)"
        x_min, y_min, x_max, y_max = result
        assert all(
            isinstance(v, (int, np.integer)) for v in (x_min, y_min, x_max, y_max)
        ), "all elements must be int"


# --- Contract: valid coarse box (is_valid_lockon_box) ---


def test_is_valid_lockon_box_valid():
    """Valid box: 4-tuple, within frame bounds, contains point."""
    frame_shape = (100, 200)  # height, width
    point = (50, 50)
    box = (10, 10, 90, 90)  # x_min, y_min, x_max, y_max
    assert is_valid_lockon_box(box, frame_shape, point) is True


def test_is_valid_lockon_box_contains_point():
    """Box must contain the click point."""
    frame_shape = (100, 200)
    point = (50, 50)
    box = (60, 60, 80, 80)  # does not contain (50, 50)
    assert is_valid_lockon_box(box, frame_shape, point) is False


def test_is_valid_lockon_box_within_bounds():
    """Box must be within frame bounds."""
    frame_shape = (100, 200)
    point = (50, 50)
    box = (0, 0, 250, 100)  # x_max out of width
    assert is_valid_lockon_box(box, frame_shape, point) is False


def test_is_valid_lockon_box_strict_bounds():
    """x_min < x_max and y_min < y_max."""
    frame_shape = (100, 200)
    point = (50, 50)
    box = (50, 50, 50, 50)  # zero-size
    assert is_valid_lockon_box(box, frame_shape, point) is False


def test_is_valid_lockon_box_none():
    """None box is invalid."""
    assert is_valid_lockon_box(None, (100, 200), (50, 50)) is False


def test_is_valid_lockon_box_wrong_length():
    """Box must be 4-tuple."""
    assert is_valid_lockon_box((1, 2, 3), (100, 200), (50, 50)) is False
    assert is_valid_lockon_box((1, 2, 3, 4, 5), (100, 200), (50, 50)) is False


def test_is_valid_lockon_box_frame_shape_3d():
    """frame_shape can be (height, width, channels)."""
    frame_shape = (100, 200, 3)
    point = (50, 50)
    box = (10, 10, 90, 90)
    assert is_valid_lockon_box(box, frame_shape, point) is True


# --- UI: Lockon button and flow (AppTest) ---


@pytest.fixture
def mock_video_processor():
    """Minimal VideoProcessor-like mock for AppTest."""
    mock = MagicMock()
    mock.width = 640
    mock.height = 480
    mock.total_frames = 10
    mock.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return mock


def test_lockon_button_present_in_coarse_selection(mock_video_processor):
    """Lockon button exists in Coarse Selection UI."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.session_state["video_processor"] = mock_video_processor
    at.session_state["current_frame"] = np.zeros((480, 640, 3), dtype=np.uint8)
    at.session_state["current_frame_idx"] = 0
    at.session_state["last_frame_idx"] = 0
    at.run()
    assert not at.exception

    # Select Coarse Selection mode
    at.radio[0].set_value("Coarse Selection").run()
    assert not at.exception

    lockon_buttons = [b for b in at.button if b.label and "Lockon" in b.label]
    assert len(lockon_buttons) >= 1, "Expected at least one button with label containing 'Lockon'"


def test_lockon_flow_sets_coarse_box_when_mock_returns_box(mock_video_processor):
    """When Lockon button is clicked and compute_lockon_box returns a box, coarse_box is set."""
    from streamlit.testing.v1 import AppTest

    # Patch where the script looks up the function (utils.lockon) so the app's import sees the mock
    with patch("utils.lockon.compute_lockon_box") as mock_compute_lockon:
        mock_compute_lockon.return_value = (50, 50, 200, 200)

        at = AppTest.from_file(str(APP_PATH))
        at.session_state["video_processor"] = mock_video_processor
        at.session_state["current_frame"] = np.zeros((480, 640, 3), dtype=np.uint8)
        at.session_state["current_frame_idx"] = 0
        at.session_state["last_frame_idx"] = 0
        at.run()
        assert not at.exception

        at.radio[0].set_value("Coarse Selection").run()
        assert not at.exception

        lockon_buttons = [b for b in at.button if b.label and "Lockon" in b.label]
        assert len(lockon_buttons) >= 1
        lockon_buttons[0].click().run()
        assert not at.exception

        # Mock must have been called when Lockon button was clicked
        assert mock_compute_lockon.called, "compute_lockon_box should be called when Lockon is clicked"
        assert at.session_state["coarse_box"] == (50, 50, 200, 200)
