"""Video processing utilities for loading and extracting frames."""
import threading
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class VideoProcessor:
    """Handle video loading and frame extraction.

    Uses a lock around cv2.VideoCapture access because OpenCV's VideoCapture
    is not thread-safe; concurrent set() and read() can cause SIGABRT in FFmpeg.
    """

    def __init__(self, video_path: str):
        """Initialize video processor.

        Args:
            video_path: Path to the MP4 video file
        """
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.current_frame_idx = 0
        self.frames = []
        self._lock = threading.Lock()
        self._load_video()

    def _load_video(self):
        """Load video file and extract metadata."""
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        with self._lock:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video.

        Args:
            frame_idx: Index of the frame to retrieve

        Returns:
            Frame as BGR numpy array or None if index is out of range
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None

        with self._lock:
            if self.cap is None:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

        if not ret:
            return None

        self.current_frame_idx = frame_idx
        return frame

    def get_all_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Extract all frames from video.

        Args:
            max_frames: Maximum number of frames to extract (None for all)

        Returns:
            List of frames as BGR numpy arrays
        """
        if self.frames:
            return self.frames

        frames = []
        with self._lock:
            if self.cap is None:
                return frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frames.append(frame)
                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    break

        self.frames = frames
        return frames
    
    def get_video_info(self) -> dict:
        """Get video metadata.
        
        Returns:
            Dictionary with video information
        """
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0
        }
    
    def close(self):
        """Release video resources."""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
