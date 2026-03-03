"""SAM-based character selection utilities."""
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
from PIL import Image
import streamlit as st


class SAMSelector:
    """Use SAM (Segment Anything Model) for intelligent character selection."""
    
    def __init__(self):
        """Initialize SAM model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM model and predictor."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Try to load model from cache or download
            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=self._get_checkpoint_path())
            sam.to(device=self.device)
            self.model = sam
            self.predictor = SamPredictor(sam)
        except Exception as e:
            st.error(f"Failed to load SAM model: {e}")
            raise
    
    def _get_checkpoint_path(self) -> str:
        """Get path to SAM checkpoint."""
        from pathlib import Path
        import urllib.request
        
        model_dir = Path.home() / ".cache" / "segment_anything"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "sam_vit_b_01ec64.pth"
        
        if not checkpoint_path.exists():
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            st.info("Downloading SAM model... This may take a few minutes.")
            urllib.request.urlretrieve(url, checkpoint_path)
        
        return str(checkpoint_path)
    
    def fuzzy_select(self, frame: np.ndarray, point: Tuple[int, int]) -> Optional[np.ndarray]:
        """Use SAM to select object from a point click (fuzzy select).
        
        Args:
            frame: Input frame as BGR numpy array
            point: (x, y) coordinate of the click
            
        Returns:
            Binary mask of selected object or None
        """
        try:
            # Convert BGR to RGB for SAM
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM
            self.predictor.set_image(frame_rgb)
            
            # Convert click point to numpy format SAM expects
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])  # 1 = foreground
            
            # Get prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Use the best mask (highest confidence)
            best_mask = masks[np.argmax(scores)]
            return best_mask.astype(np.uint8) * 255
        except Exception as e:
            st.error(f"Error in fuzzy select: {e}")
            return None
    
    def refine_coarse_selection(self, frame: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """Refine a coarse selection by removing background parts.
        
        Args:
            frame: Input frame as BGR numpy array
            coarse_mask: Coarse binary mask from cursor selection
            
        Returns:
            Refined binary mask
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(frame_rgb)
            
            # Get coordinates of selected pixels
            selected_coords = np.argwhere(coarse_mask > 128)
            
            if len(selected_coords) == 0:
                return coarse_mask
            
            # Get center point from selected region
            center_y, center_x = selected_coords.mean(axis=0).astype(int)
            
            # Use SAM to refine
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Use best mask
            refined_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
            
            # Combine with original coarse selection
            combined = cv2.bitwise_and(coarse_mask, refined_mask)
            
            return combined
        except Exception as e:
            st.warning(f"Could not refine selection: {e}. Using coarse selection.")
            return coarse_mask
