from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "lidc_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing
VOX_MM = (1.0, 1.0, 1.0)      # spacing as 1x1x1
CUBE = (48, 48, 48)           # if things dont work at all, make it 32x32x32
MIN_VOXELS = 10               # skip tiny nodules
NEG_PER_POS = 1   
# get negatives from annotations with no scans
NEG_FROM_EMPTY_SCAN = 2      # amount of cubes to get from unannotated
MIN_BODY_INTENSITY = 0.05    # ignore air cubes
MAX_EMPTY_TRIES_PER_SCAN = 200  # 
            
VAL_SPLIT = 0.2
SEED = 1337

# Training
BATCH = 8                     # lower if you hit OOM
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
