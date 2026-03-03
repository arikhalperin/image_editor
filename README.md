# Video Frame Editor with AI Background Removal

A Python Streamlit application that allows you to load MP4 videos, navigate frame by frame, select characters using AI, and remove backgrounds intelligently.

## Features

✨ **Video Processing**
- Load and play MP4 videos
- Extract frames automatically
- Frame-by-frame navigation with slider

🎯 **Two Selection Modes**
1. **Fuzzy Select** - Uses SAM (Segment Anything Model) for intelligent object detection
2. **Coarse Selection** - Manual cursor-based selection with AI refinement to exclude background

🧠 **AI-Powered Background Removal**
- Intelligent background removal using REMBG
- Separate foreground (character) from background
- Export edited frames

## Installation

1. Clone the repository:
```bash
cd image_editor
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Upload an MP4 video file
2. Navigate through frames using the slider
3. Choose a selection mode (Fuzzy or Coarse)
4. Select the character in the frame
5. Remove the background
6. Download the edited frame

## Project Structure

```
image_editor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── utils/
    ├── __init__.py
    ├── video_processor.py # Video loading and frame extraction
    ├── sam_selector.py    # SAM-based character selection
    └── background_remover.py # REMBG background removal
```

## Technical Stack

- **Streamlit** - Web UI framework
- **OpenCV** - Video processing
- **SAM** - Segment Anything Model for object detection
- **REMBG** - AI-powered background removal
- **PyTorch** - Deep learning framework
- **NumPy/PIL** - Image processing

## Performance Notes

- First run downloads SAM model (~350MB) and REMBG model (~300MB)
- Coarse selection refines user input to handle background parts intelligently
- Background removal uses REMBG's pre-trained model
