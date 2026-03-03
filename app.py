"""Main Streamlit application for video frame editor with AI background removal."""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.video_processor import VideoProcessor
from utils.sam_selector import SAMSelector
from utils.background_remover import BackgroundRemover
from utils.lockon import compute_lockon_box


# Page configuration
st.set_page_config(
    page_title="Video Frame Editor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Video Frame Editor with AI Background Removal")
st.markdown("Extract frames, select characters, and remove backgrounds with AI intelligence.")

# Initialize session state
if "video_processor" not in st.session_state:
    st.session_state.video_processor = None
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "selection_mask" not in st.session_state:
    st.session_state.selection_mask = None
if "edited_frame" not in st.session_state:
    st.session_state.edited_frame = None
if "sam_selector" not in st.session_state:
    st.session_state.sam_selector = None
if "bg_remover" not in st.session_state:
    st.session_state.bg_remover = None
if "drawing" not in st.session_state:
    st.session_state.drawing = False
if "brush_points" not in st.session_state:
    st.session_state.brush_points = []
if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = None
if "coarse_box" not in st.session_state:
    st.session_state.coarse_box = None  # Store (x_min, y_min, x_max, y_max)
if "fuzzy_selections" not in st.session_state:
    st.session_state.fuzzy_selections = []  # List of fuzzy selection masks
if "fuzzy_click_points" not in st.session_state:
    st.session_state.fuzzy_click_points = []  # List of (x, y) click coordinates for each selection
if "combined_mask" not in st.session_state:
    st.session_state.combined_mask = None  # Combined result of all selections
if "fuzzy_click_pending" not in st.session_state:
    st.session_state.fuzzy_click_pending = None  # Pending (x, y) for sliders from image click
if "last_frame_idx" not in st.session_state:
    st.session_state.last_frame_idx = None
if "lockon_click_pending" not in st.session_state:
    st.session_state.lockon_click_pending = None  # (x, y) click on character for lockon
if "lockon_preview_key" not in st.session_state:
    st.session_state.lockon_preview_key = 0  # increment after Lockon so next click registers


# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Video upload
    st.subheader("1. Load Video")
    video_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
    
    if video_file is not None:
        # Save uploaded file temporarily
        video_path = f"/tmp/{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Load video
        try:
            if st.session_state.video_processor is None:
                with st.spinner("Loading video..."):
                    st.session_state.video_processor = VideoProcessor(video_path)
                    st.session_state.current_frame_idx = 0
                    st.session_state.selection_mask = None
                    st.session_state.edited_frame = None
                    st.session_state.brush_points = []
            
            video_info = st.session_state.video_processor.get_video_info()
            st.success(f"✅ Video loaded: {video_info['total_frames']} frames @ {video_info['fps']:.1f} FPS")
            st.text(f"Resolution: {video_info['width']}x{video_info['height']}")
            st.text(f"Duration: {video_info['duration_seconds']:.1f}s")
        except Exception as e:
            st.error(f"Error loading video: {e}")
    else:
        st.info("👈 Upload a video file in the sidebar to start")


# Main content
if st.session_state.video_processor is not None:
    # Frame navigation
    st.subheader("2. Navigate Frames")
    
    total_frames = st.session_state.video_processor.total_frames
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_idx = st.slider(
            "Select frame",
            min_value=0,
            max_value=total_frames - 1,
            value=st.session_state.current_frame_idx,
            key="frame_slider"
        )
    
    with col2:
        st.metric("Frame", f"{frame_idx + 1}/{total_frames}")
    
    if st.session_state.last_frame_idx is None:
        st.session_state.last_frame_idx = frame_idx

    if frame_idx != st.session_state.last_frame_idx:
        # New frame: clear fuzzy selections but keep coarse selection.
        st.session_state.fuzzy_selections = []
        st.session_state.fuzzy_click_points = []
        st.session_state.combined_mask = None
        st.session_state.lockon_click_pending = None
        if st.session_state.selection_mode == "fuzzy":
            st.session_state.selection_mask = None
            st.session_state.selection_mode = None
        st.session_state.last_frame_idx = frame_idx

    st.session_state.current_frame_idx = frame_idx
    
    # Get current frame
    current_frame = st.session_state.video_processor.get_frame(frame_idx)
    if current_frame is not None:
        st.session_state.current_frame = current_frame
    
    if st.session_state.current_frame is not None:
        # Character selection mode
        st.subheader("3. Select Character")
        
        selection_mode = st.radio(
            "Selection mode",
            ["View", "Fuzzy Select", "Coarse Selection"],
            horizontal=True
        )
        
        # Display the frame with selection UI
        # Always make a fresh copy to avoid modifying the original frame
        frame_display = st.session_state.current_frame.copy()
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB).copy()

        # Show current selection mask overlay
        if st.session_state.selection_mask is not None:
            mask_display = st.session_state.selection_mask.copy()
            if len(mask_display.shape) == 2:
                mask_color = np.zeros_like(frame_rgb)
                mask_color[:, :, 2] = (mask_display * 0.5).astype(np.uint8)  # Red overlay
                # Create overlay without modifying original
                overlay_rgb = cv2.addWeighted(frame_rgb.copy(), 1, mask_color, 0.5, 0)
                frame_rgb = overlay_rgb

        # Display frame and handle clicks
        if selection_mode == "Fuzzy Select":
            st.markdown("**Add fuzzy selections by entering coordinates**")
            
            # Initialize SAM if needed
            if st.session_state.sam_selector is None:
                with st.spinner("Loading AI model..."):
                    st.session_state.sam_selector = SAMSelector()
            
            # Apply pending click before widget instantiation to avoid StreamlitAPIException
            if st.session_state.fuzzy_click_pending is not None:
                pending_x, pending_y = st.session_state.fuzzy_click_pending
                st.session_state.fuzzy_x_slider = int(pending_x)
                st.session_state.fuzzy_y_slider = int(pending_y)
                st.session_state.fuzzy_click_pending = None

            if "fuzzy_x_slider" not in st.session_state:
                st.session_state.fuzzy_x_slider = st.session_state.video_processor.width // 2
            if "fuzzy_y_slider" not in st.session_state:
                st.session_state.fuzzy_y_slider = st.session_state.video_processor.height // 2

            # Define position sliders BEFORE columns so they can be used in both
            st.markdown("**Point Position** - Click or drag sliders to position the yellow crosshair")
            col_slider_x, col_slider_y = st.columns(2)
            
            with col_slider_x:
                x = st.slider("X position", 0, st.session_state.video_processor.width - 1, value=st.session_state.video_processor.width // 2, key="fuzzy_x_slider")
            
            with col_slider_y:
                y = st.slider("Y position", 0, st.session_state.video_processor.height - 1, value=st.session_state.video_processor.height // 2, key="fuzzy_y_slider")
            
            st.caption(f"📍 Current crosshair position: ({int(x)}, {int(y)})")
            
            # Create two-column layout: Image on left, Controls on right
            col_image, col_controls = st.columns([3, 1])
            
            # LEFT COLUMN: Image display with coarse box
            with col_image:
                # Create display frame with coarse box overlay - use fresh copy
                frame_display = st.session_state.current_frame.copy()
                frame_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)

                # Draw coarse box if it exists
                if st.session_state.coarse_box is not None:
                    x_min, y_min, x_max, y_max = st.session_state.coarse_box
                    cv2.rectangle(frame_display_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame_display_rgb, "Coarse Box", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show overlay with current selections if any
                if st.session_state.fuzzy_selections:
                    combined = np.zeros_like(frame_display_rgb[:, :, 0])
                    for sel_mask in st.session_state.fuzzy_selections:
                        combined = cv2.bitwise_or(combined, sel_mask)
                    
                    # Create red overlay for selections
                    overlay = frame_display_rgb.copy()
                    overlay[combined > 128] = [255, 0, 0]  # Red for selected areas
                    frame_display_rgb = cv2.addWeighted(frame_display_rgb, 0.7, overlay, 0.3, 0)

                # Draw click points with labels showing which selection they belong to
                for idx, (click_x, click_y) in enumerate(st.session_state.fuzzy_click_points, 1):
                    # Draw circle at click point
                    cv2.circle(frame_display_rgb, (int(click_x), int(click_y)), 8, (0, 255, 255), 2)  # Cyan circle
                    # Draw label
                    cv2.putText(frame_display_rgb, f"S{idx}", (int(click_x) + 10, int(click_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw current crosshair/cursor position (from sliders above)
                cursor_x, cursor_y = int(x), int(y)
                # Draw crosshair
                line_length = 20
                cv2.line(frame_display_rgb, (cursor_x - line_length, cursor_y), (cursor_x + line_length, cursor_y), (255, 255, 0), 1)  # Yellow crosshair
                cv2.line(frame_display_rgb, (cursor_x, cursor_y - line_length), (cursor_x, cursor_y + line_length), (255, 255, 0), 1)
                cv2.circle(frame_display_rgb, (cursor_x, cursor_y), 5, (255, 255, 0), 2)  # Yellow dot in center

                st.caption("Fuzzy Selections (yellow crosshair = click point, cyan circles = previous selections) - **Click on image to set position**")
                
                # Use interactive image that captures clicks
                coords = streamlit_image_coordinates(
                    Image.fromarray(frame_display_rgb)
                )
                
                # Update sliders if user clicked on image
                if coords is not None:
                    st.session_state.fuzzy_click_pending = (int(coords["x"]), int(coords["y"]))
                    st.rerun()
                else:
                    # Help text for positioning if no click
                    st.caption("👆 **Click on image to set position, or use sliders above**")
            
            # RIGHT COLUMN: Controls and selections list
            with col_controls:
                st.subheader("🎯 Selections")
                
                if st.session_state.coarse_box is not None:
                    x_min, y_min, x_max, y_max = st.session_state.coarse_box
                    st.caption(f"Box: ({x_min},{y_min})-({x_max},{y_max})")
                
                if st.button("➕ Add Selection", key="add_fuzzy"):
                    with st.spinner("Analyzing..."):
                        mask = st.session_state.sam_selector.fuzzy_select(
                            st.session_state.current_frame,
                            (int(x), int(y))
                        )
                        if mask is not None:
                            # If coarse box exists, apply it to the mask
                            if st.session_state.coarse_box is not None:
                                x_min, y_min, x_max, y_max = st.session_state.coarse_box
                                coarse_mask = np.zeros_like(mask)
                                coarse_mask[y_min:y_max, x_min:x_max] = 255
                                mask = cv2.bitwise_and(mask, coarse_mask)
                            
                            # Add to fuzzy selections list
                            st.session_state.fuzzy_selections.append(mask)
                            st.session_state.fuzzy_click_points.append((int(x), int(y)))  # Track click point
                            
                            # Update combined mask (accumulate all selections)
                            combined = np.zeros_like(mask)
                            for sel_mask in st.session_state.fuzzy_selections:
                                combined = cv2.bitwise_or(combined, sel_mask)
                            st.session_state.combined_mask = combined
                            st.session_state.selection_mode = "fuzzy"
                            st.success(f"✅ Selection {len(st.session_state.fuzzy_selections)} added!")
                            st.rerun()
                
                # Selections list
                st.markdown("---")
                st.markdown("**Selections List**")
                
                if st.session_state.fuzzy_selections:
                    for idx, sel_mask in enumerate(st.session_state.fuzzy_selections, 1):
                        pixels = np.sum(sel_mask > 128)
                        click_x, click_y = st.session_state.fuzzy_click_points[idx - 1]
                        
                        with st.container(border=True):
                            col_thumb, col_info, col_delete = st.columns([1, 2, 0.5])
                            
                            # Show thumbnail in a small container
                            with col_thumb:
                                # Create a thumbnail by cropping the selection
                                y_coords, x_coords = np.where(sel_mask > 128)
                                if len(y_coords) > 0:
                                    y_min_sel, y_max_sel = y_coords.min(), y_coords.max()
                                    x_min_sel, x_max_sel = x_coords.min(), x_coords.max()
                                    
                                    # Add padding
                                    pad = 5
                                    y_min_sel = max(0, y_min_sel - pad)
                                    y_max_sel = min(st.session_state.current_frame.shape[0], y_max_sel + pad)
                                    x_min_sel = max(0, x_min_sel - pad)
                                    x_max_sel = min(st.session_state.current_frame.shape[1], x_max_sel + pad)
                                    
                                    # Create cropped mask thumbnail
                                    crop_mask = sel_mask[y_min_sel:y_max_sel, x_min_sel:x_max_sel]
                                    crop_frame = frame_rgb[y_min_sel:y_max_sel, x_min_sel:x_max_sel]
                                    
                                    # Show where the mask is
                                    if crop_mask.shape[0] > 10 and crop_mask.shape[1] > 10:
                                        st.image(crop_mask, width=60)
                            
                            with col_info:
                                st.markdown(f"**S{idx}**")
                                st.caption(f"Pixels: {pixels:,}")
                                st.caption(f"Click: ({click_x}, {click_y})")
                            
                            with col_delete:
                                if st.button("🗑️", key=f"del_sel_{idx}"):
                                    st.session_state.fuzzy_selections.pop(idx - 1)
                                    st.session_state.fuzzy_click_points.pop(idx - 1)
                                    if st.session_state.fuzzy_selections:
                                        combined = np.zeros_like(st.session_state.fuzzy_selections[0])
                                        for sel_mask in st.session_state.fuzzy_selections:
                                            combined = cv2.bitwise_or(combined, sel_mask)
                                        st.session_state.combined_mask = combined
                                    else:
                                        st.session_state.combined_mask = None
                                    st.rerun()
                    
                    st.markdown("---")
                    col_apply, col_clear = st.columns(2)
                    
                    with col_apply:
                        if st.button("✅ Apply All", key="apply_all"):
                            st.session_state.selection_mask = st.session_state.combined_mask
                            st.success("✅ Applied!")
                            st.rerun()
                    
                    with col_clear:
                        if st.button("❌ Clear All", key="clear_all"):
                            st.session_state.fuzzy_selections = []
                            st.session_state.fuzzy_click_points = []
                            st.session_state.combined_mask = None
                            st.rerun()
        
        elif selection_mode == "Coarse Selection":
            st.markdown("**Define rectangular area around the character**")

            # Sync sliders to coarse_box before widgets are created (e.g. after Lockon)
            if st.session_state.get("sync_sliders_to_coarse_box") and st.session_state.coarse_box is not None:
                b = st.session_state.coarse_box
                st.session_state["coarse_x_start"] = b[0]
                st.session_state["coarse_y_start"] = b[1]
                st.session_state["coarse_x_end"] = b[2]
                st.session_state["coarse_y_end"] = b[3]
                st.session_state["sync_sliders_to_coarse_box"] = False

            # Create helper text
            st.info("📍 Use the sliders below to define the rectangular selection area")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_start = st.slider(
                    "X Start", 
                    0, 
                    st.session_state.video_processor.width - 1, 
                    value=st.session_state.video_processor.width // 4,
                    key="coarse_x_start"
                )
                y_start = st.slider(
                    "Y Start", 
                    0, 
                    st.session_state.video_processor.height - 1, 
                    value=st.session_state.video_processor.height // 4,
                    key="coarse_y_start"
                )
            
            with col2:
                x_end = st.slider(
                    "X End", 
                    0, 
                    st.session_state.video_processor.width - 1, 
                    value=st.session_state.video_processor.width * 3 // 4,
                    key="coarse_x_end"
                )
                y_end = st.slider(
                    "Y End", 
                    0, 
                    st.session_state.video_processor.height - 1, 
                    value=st.session_state.video_processor.height * 3 // 4,
                    key="coarse_y_end"
                )

            # Create preview with rectangle drawn
            frame_preview = frame_rgb.copy()
            x_min, x_max = min(int(x_start), int(x_end)), max(int(x_start), int(x_end))
            y_min, y_max = min(int(y_start), int(y_end)), max(int(y_start), int(y_end))

            # Always draw the green rectangle from sliders so moving sliders updates the box (single source of truth)
            cv2.rectangle(frame_preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            if st.session_state.coarse_box is not None:
                cv2.putText(frame_preview, "Coarse selection", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw lockon click point if set
            if st.session_state.lockon_click_pending is not None:
                lx, ly = st.session_state.lockon_click_pending
                cv2.circle(frame_preview, (int(lx), int(ly)), 8, (0, 255, 255), 2)
                cv2.putText(frame_preview, "Lockon", (int(lx) + 10, int(ly)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            st.caption("**Lockon:** 1) Click on the character in the image below. 2) Press the **Lockon** button. The green frame will update to the character. Click a new spot and press Lockon again to retry.")
            coords_coarse = streamlit_image_coordinates(Image.fromarray(frame_preview), key=f"coarse_preview_{st.session_state.lockon_preview_key}")
            if coords_coarse is not None:
                new_point = (int(coords_coarse["x"]), int(coords_coarse["y"]))
                # Only rerun when the click is new so the Lockon button can run (component may re-emit same coords)
                if new_point != st.session_state.lockon_click_pending:
                    st.session_state.lockon_click_pending = new_point
                    st.rerun()

            if st.session_state.lockon_click_pending is not None:
                lx, ly = st.session_state.lockon_click_pending
                st.info(f"Point set at ({lx}, {ly}). Press **Lockon** to zoom on character (or click elsewhere and Lockon again to retry).")

            col_buttons1, col_buttons2, col_buttons3, col_buttons4 = st.columns(4)
            
            with col_buttons1:
                if st.button("📦 Set Coarse Box"):
                    # Remember the coarse box
                    st.session_state.coarse_box = (x_min, y_min, x_max, y_max)
                    # Reset fuzzy selections for new box
                    st.session_state.fuzzy_selections = []
                    st.session_state.fuzzy_click_points = []
                    st.session_state.combined_mask = None
                    st.success("✅ Coarse box set! Switch to Fuzzy Select mode to add selections.")
                    st.rerun()
            
            with col_buttons2:
                if st.button("🎨 Use Box as Mask"):
                    # Create mask from rectangular area
                    coarse_mask = np.zeros((st.session_state.video_processor.height, st.session_state.video_processor.width), dtype=np.uint8)
                    coarse_mask[y_min:y_max, x_min:x_max] = 255
                    
                    st.session_state.selection_mask = coarse_mask
                    st.session_state.coarse_box = (x_min, y_min, x_max, y_max)
                    st.session_state.selection_mode = "coarse"
                    st.success("✅ Coarse mask created!")
                    st.rerun()
            
            with col_buttons3:
                if st.button("🗑️ Clear Box"):
                    st.session_state.coarse_box = None
                    st.session_state["sync_sliders_to_coarse_box"] = False
                    st.session_state.fuzzy_selections = []
                    st.session_state.fuzzy_click_points = []
                    st.session_state.combined_mask = None
                    st.session_state.selection_mask = None
                    st.session_state.selection_mode = None
                    st.session_state.lockon_click_pending = None
                    st.rerun()

            with col_buttons4:
                lockon_label = "Lockon" if st.session_state.coarse_box is None else "Lockon (try again)"
                if st.button(lockon_label, key="lockon_btn"):
                    point = st.session_state.lockon_click_pending
                    if point is None:
                        point = (st.session_state.video_processor.width // 2, st.session_state.video_processor.height // 2)
                        st.warning("No click yet — using frame center. Click on the character and press Lockon again for better results.")
                    if st.session_state.sam_selector is None:
                        with st.spinner("Loading AI model..."):
                            st.session_state.sam_selector = SAMSelector()
                    with st.spinner("Running Lockon..."):
                        box = compute_lockon_box(
                            st.session_state.current_frame,
                            point,
                            selector=st.session_state.sam_selector,
                        )
                    if box is not None:
                        st.session_state.coarse_box = box
                        # Sync sliders on next run (cannot set widget keys after they're instantiated)
                        st.session_state["sync_sliders_to_coarse_box"] = True
                        st.session_state.fuzzy_selections = []
                        st.session_state.fuzzy_click_points = []
                        st.session_state.combined_mask = None
                        st.session_state.lockon_preview_key += 1  # fresh image so next click registers
                        st.success("✅ Lockon set coarse box! The green frame should now wrap the character. Click a new spot and press Lockon again to retry.")
                    else:
                        st.error("Lockon failed. Try clicking directly on the character and press Lockon again.")
                    st.rerun()
            
            if st.session_state.selection_mask is not None:
                col_refine, col_space = st.columns([1, 1])
                
                with col_refine:
                    if st.button("🔄 Refine with AI"):
                        if st.session_state.sam_selector is None:
                            with st.spinner("Loading AI model..."):
                                st.session_state.sam_selector = SAMSelector()
                        
                        with st.spinner("Refining selection..."):
                            refined_mask = st.session_state.sam_selector.refine_coarse_selection(
                                st.session_state.current_frame,
                                st.session_state.selection_mask
                            )
                            st.session_state.selection_mask = refined_mask
                            st.success("✅ Selection refined!")
                            st.rerun()
        
        else:  # View mode
            st.image(frame_rgb, caption=f"Frame {frame_idx + 1}/{total_frames}")
        
        # Background removal
        st.subheader("4. Remove Background")
        
        # Use combined mask if available, otherwise use selection_mask
        active_mask = st.session_state.combined_mask if st.session_state.combined_mask is not None else st.session_state.selection_mask
        
        if active_mask is not None:
            col_remove, col_export = st.columns(2)
            
            with col_remove:
                if st.button("🎨 Remove Background"):
                    with st.spinner("Removing background..."):
                        if st.session_state.fuzzy_selections:  # If fuzzy selections exist
                            # For fuzzy select: keep ONLY the selected area
                            if st.session_state.bg_remover is None:
                                st.session_state.bg_remover = BackgroundRemover()
                            edited = st.session_state.bg_remover.apply_selection_mask(
                                st.session_state.current_frame,
                                active_mask,
                                alpha=1.0
                            )
                        else:
                            # For coarse selection: use smart AI removal
                            if st.session_state.bg_remover is None:
                                st.session_state.bg_remover = BackgroundRemover()
                            
                            edited = st.session_state.bg_remover.remove_background(
                                st.session_state.current_frame,
                                active_mask
                            )
                        st.session_state.edited_frame = edited
                        st.success("✅ Background removed!")
            
            with col_export:
                if st.session_state.edited_frame is not None:
                    st.markdown("**Export Settings**")
                    
                    # Get current frame dimensions
                    current_h, current_w = st.session_state.edited_frame.shape[:2]
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        export_width = st.number_input(
                            "Export Width (px)",
                            min_value=100,
                            max_value=4000,
                            value=current_w,
                            key="export_width"
                        )
                    
                    with export_col2:
                        export_height = st.number_input(
                            "Export Height (px)",
                            min_value=100,
                            max_value=4000,
                            value=current_h,
                            key="export_height"
                        )
                    
                    export_scale = st.number_input(
                        "Scale Factor",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key="export_scale",
                        help="1.0 = original size, <1.0 = shrink, >1.0 = enlarge"
                    )
                    
                    if st.button("💾 Export Frame"):
                        output_path = f"edited_frame_{frame_idx}.png"
                        if st.session_state.bg_remover.export_frame(
                            st.session_state.edited_frame,
                            output_path,
                            canvas_width=int(export_width),
                            canvas_height=int(export_height),
                            scale_factor=export_scale
                        ):
                            st.success(f"✅ Frame exported to {output_path} ({int(export_width)}x{int(export_height)}) at {export_scale:.1f}x scale")
        else:
            st.info("👆 Select a character first to remove background")
        
        # Display results
        if st.session_state.edited_frame is not None:
            st.subheader("5. Result")
            
            # Display with transparency (keep RGBA as is)
            st.image(st.session_state.edited_frame, caption="Edited Frame")
            
            # Download button
            png_path = f"edited_frame_{frame_idx}.png"
            if Path(png_path).exists():
                with open(png_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Edited Frame",
                        data=f.read(),
                        file_name=png_path,
                        mime="image/png"
                    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎬 Video Frame Editor with AI Background Removal</p>
        <p style='font-size: 0.8em; color: gray;'>Powered by SAM and REMBG</p>
    </div>
    """,
    unsafe_allow_html=True
)
