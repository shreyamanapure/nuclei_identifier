from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'img1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'img2.png'

# ML Model config
# MODEL_DIR = ROOT / 'weights'
SEGMENTATION_MODEL = 'best.pt'
# NERVE_MODEL = MODEL_DIR / 'best_nerve.pt'