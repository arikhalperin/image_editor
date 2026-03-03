"""Background removal utilities using REMBG."""
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import streamlit as st


class BackgroundRemover:
    """Remove background from images using REMBG."""
    
    def __init__(self):
        """Initialize background remover."""
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load REMBG model."""
        try:
            from rembg import new_session
            self.session = new_session(model_name="u2net")
        except Exception as e:
            st.error(f"Failed to load REMBG model: {e}")
            raise
    
    def remove_background(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Remove background from frame, optionally using a selection mask.
        
        Args:
            frame: Input frame as BGR numpy array
            mask: Optional binary mask for selected region (foreground)
            
        Returns:
            RGBA image with background removed (4 channels)
        """
        try:
            from rembg import remove
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply REMBG
            output = remove(
                Image.fromarray(frame_rgb),
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10
            )
            
            # Convert back to numpy
            output_array = np.array(output)
            
            # If mask provided, use it to further refine
            if mask is not None:
                # Ensure mask is single channel
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                # Normalize mask to 0-1
                mask_normalized = mask.astype(np.float32) / 255.0
                
                # Apply mask to alpha channel
                if output_array.shape[2] == 4:  # RGBA
                    output_array[:, :, 3] = (output_array[:, :, 3].astype(np.float32) * mask_normalized).astype(np.uint8)
            
            return output_array
        except Exception as e:
            st.error(f"Error removing background: {e}")
            return frame
    
    def apply_selection_mask(self, frame: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Apply selection mask to frame and remove background.
        
        Args:
            frame: Input frame as BGR numpy array
            mask: Binary mask of selected area (0-255)
            alpha: Alpha blending factor (0-1)
            
        Returns:
            Frame with background removed based on mask (RGBA)
        """
        try:
            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Normalize mask to 0-1 range
            mask_float = (mask.astype(np.float32) / 255.0) * alpha
            
            # Convert BGR frame to RGBA
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            
            # Apply mask to alpha channel
            frame_rgba[:, :, 3] = (mask_float * 255).astype(np.uint8)
            
            # Zero out RGB values where alpha is 0 (for truly transparent areas)
            # Expand mask to 3 dimensions (H, W, 3)
            mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
            frame_rgba[:, :, :3] = (frame_rgba[:, :, :3].astype(np.float32) * mask_3d).astype(np.uint8)
            
            return frame_rgba
        except Exception as e:
            st.error(f"Error applying mask: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
    def export_frame(self, frame: np.ndarray, output_path: str, canvas_width: Optional[int] = None, canvas_height: Optional[int] = None, scale_factor: float = 1.0):
        """Export frame as PNG with transparency, optionally on a canvas with scaling.
        
        Args:
            frame: RGBA image array
            output_path: Path to save the image
            canvas_width: Width of the output canvas (if None, uses frame width)
            canvas_height: Height of the output canvas (if None, uses frame height)
            scale_factor: Scale factor to apply (1.0 = original, <1.0 = shrink, >1.0 = enlarge)
        """
        try:
            frame_h, frame_w = frame.shape[:2]
            
            # Apply scale factor
            scaled_w = int(frame_w * scale_factor)
            scaled_h = int(frame_h * scale_factor)
            
            if scale_factor != 1.0:
                # Resize frame
                if scale_factor < 1.0:
                    scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                else:
                    scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_frame = frame
            
            if canvas_width is not None and canvas_height is not None:
                # Create canvas
                canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
                
                # Find the bounding box of non-transparent pixels (alpha channel)
                alpha_channel = scaled_frame[:, :, 3]
                non_transparent = np.where(alpha_channel > 0)

                if len(non_transparent[0]) > 0:
                    # Find bounds of actual content
                    top_pixel = non_transparent[0].min()
                    bottom_pixel = non_transparent[0].max()
                    left_pixel = non_transparent[1].min()
                    right_pixel = non_transparent[1].max()

                    # Calculate center of content
                    content_center_y = (top_pixel + bottom_pixel) // 2
                    content_center_x = (left_pixel + right_pixel) // 2

                    # Calculate where to place the frame so content is centered
                    canvas_center_x = canvas_width // 2
                    canvas_center_y = canvas_height // 2

                    # Offset to center the content
                    x_offset = canvas_center_x - content_center_x
                    y_offset = canvas_center_y - content_center_y
                else:
                    # No non-transparent pixels, center the entire frame
                    scaled_h_actual, scaled_w_actual = scaled_frame.shape[:2]
                    x_offset = (canvas_width - scaled_w_actual) // 2
                    y_offset = (canvas_height - scaled_h_actual) // 2

                # Place frame on canvas with calculated offset
                scaled_h_actual, scaled_w_actual = scaled_frame.shape[:2]

                # Determine which parts of the source and canvas to use
                src_y_start = max(0, -y_offset)
                src_x_start = max(0, -x_offset)
                canvas_y_start = max(0, y_offset)
                canvas_x_start = max(0, x_offset)

                src_y_end = min(scaled_h_actual, canvas_height - canvas_y_start + src_y_start)
                src_x_end = min(scaled_w_actual, canvas_width - canvas_x_start + src_x_start)

                canvas_y_end = canvas_y_start + (src_y_end - src_y_start)
                canvas_x_end = canvas_x_start + (src_x_end - src_x_start)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = \
                        scaled_frame[src_y_start:src_y_end, src_x_start:src_x_end]

                frame_to_export = canvas
            else:
                frame_to_export = scaled_frame
            
            # Convert from RGBA to BGRA for OpenCV
            if frame_to_export.shape[2] == 4:
                frame_bgra = cv2.cvtColor(frame_to_export, cv2.COLOR_RGBA2BGRA)
            else:
                frame_bgra = frame_to_export
            
            cv2.imwrite(output_path, frame_bgra)
            return True
        except Exception as e:
            st.error(f"Error exporting frame: {e}")
            return False
