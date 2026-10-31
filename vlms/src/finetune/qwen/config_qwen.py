"""Shared configuration for the CV4DC drawing-render verification pipeline.

Used by train_qwen.py, eval_qwen.py and attributes_qwen.py. The data scripts
(build_manifest.py, prepare_data.py) read config.py; keep the data paths in sync.
"""

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"   


SYSTEM_PROMPT = (
    "You are an expert mechanical engineer performing quality inspection "
    "of 3D-printed upper-limb prosthetic components."
)

QUESTION = (
    "You are shown two images of a mechanical part. The first image is a 2D "
    "engineering drawing. The second image is a shaded render of a 3D CAD model. "
    "The two images may show the part from different viewpoints. "
    "Do they depict the same part? "
    "Answer with exactly one word: Yes or No."
)


YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO"]


MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28   

SEED = 13

# BlueCAD-30K source folders
BLUECAD_ROOT = ""
RENDERS_ROOT = f"{BLUECAD_ROOT}/renders"
DRAWINGS_ROOT = f"{BLUECAD_ROOT}/drawings"


RENDER_AXIS_TO_VIEW = {"+x": "right", "-x": "left", "+y": "back",
                       "-y": "front", "+z": "top", "-z": "bottom"}

RENDER_VIEW_REMAP = {}

RENDER_INDEX_TO_VIEW = None

DRAWING_VARIANT = "normal"

# Generated files
DATA_DIR = ""
MANIFEST_CSV = f"{DATA_DIR}/manifest.csv"
TRAIN_JSONL = f"{DATA_DIR}/train_{DRAWING_VARIANT}.jsonl"
VAL_JSONL = f"{DATA_DIR}/val_{DRAWING_VARIANT}.jsonl"
TEST_JSONL = "data/test.jsonl"           

NEG_PER_POS = 1          
HARD_NEG_FRACTION = 0.7  

#  LoRA 
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv", "proj", "linear_fc1", "linear_fc2",
]

#  train 
OUTPUT_DIR = "runs/qwen3vl8b_lora_bluecad30k"
LEARNING_RATE = 1e-4
EPOCHS = 1
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16          
WARMUP_RATIO = 0.03      
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250
MAX_TRAIN_SAMPLES = None  