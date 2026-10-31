"""Shared configuration for the CV4DC drawing-render verification pipeline.

Used by train_qwen.py, eval_qwen.py and attributes_qwen.py. The data scripts
(build_manifest.py, prepare_data.py) read config.py; keep the data paths in sync.
"""

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"   # Qwen/Qwen3-VL-2B-Instruct for a cheaper comparison row

# ---------------------------------------------------------------- prompt ----
# Keep this EXACTLY identical between zero-shot eval, training, and fine-tuned
# eval, otherwise the calibration comparison is confounded.
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

# Answer vocabulary. We aggregate log-probs over these surface forms because
# tokenizers may split "Yes"/" Yes"/"yes" differently.
YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO"]

# ---------------------------------------------------------------- images ----
# Cap vision tokens to keep memory predictable on a consumer card.
# Qwen's image processor works on pixel budgets (28x28 patches).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28   # raise to 1536*28*28 if drawings' small text matters and VRAM allows

# ---------------------------------------------------------------- data ------
SEED = 13

# BlueCAD-30K source folders
BLUECAD_ROOT = "/home/ec2-user/cad-2d-vision-foundational/BlueCAD30K/data"
RENDERS_ROOT = f"{BLUECAD_ROOT}/renders"
DRAWINGS_ROOT = f"{BLUECAD_ROOT}/drawings"

# Renders are numbered (000.png ... 005.png); their views come from the camera
# poses in each part's transforms.json. A camera on the +x side of the part shows
# the "right" view, etc. (paper Table 10, native STEP coordinates).
RENDER_AXIS_TO_VIEW = {"+x": "right", "-x": "left", "+y": "back",
                       "-y": "front", "+z": "top", "-z": "bottom"}
# If build_manifest.py's silhouette check reports a mismatch, paste the remap it
# suggests here, e.g. {"top": "back", "back": "bottom", ...}.
RENDER_VIEW_REMAP = {}
# Last resort for parts with no usable transforms.json: list of views in file
# order (000, 001, ...). None = skip such parts instead of guessing.
RENDER_INDEX_TO_VIEW = None

# Which drawings to train on: "normal" (clean), "dim" (dimensioned) or "mix"
DRAWING_VARIANT = "normal"

# Generated files
DATA_DIR = "/home/ec2-user/cad-2d-vlm-accv/scripts/finetune/data/bluecad30k"
MANIFEST_CSV = f"{DATA_DIR}/manifest.csv"
TRAIN_JSONL = f"{DATA_DIR}/train_{DRAWING_VARIANT}.jsonl"
VAL_JSONL = f"{DATA_DIR}/val_{DRAWING_VARIANT}.jsonl"
TEST_JSONL = "data/test.jsonl"            # the 1,000-pair benchmark (its parts are excluded from train/val)
FUSION360_JSONL = "data/fusion360.jsonl"  # ~300 held-out generalization pairs

# Negative sampling mix per positive pair
NEG_PER_POS = 1          # keep classes balanced: 1 negative per positive
HARD_NEG_FRACTION = 0.7  # fraction of negatives that are geometry-aware hard negatives

# ---------------------------------------------------------------- LoRA ------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv", "proj", "linear_fc1", "linear_fc2",
]

# ---------------------------------------------------------------- train -----
OUTPUT_DIR = "runs/qwen3vl8b_lora_bluecad30k"
LEARNING_RATE = 1e-4
EPOCHS = 1
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16          # effective batch 16
WARMUP_RATIO = 0.03      # a ratio: pass as warmup_ratio, not warmup_steps
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10
SAVE_STEPS = 250
EVAL_STEPS = 250
MAX_TRAIN_SAMPLES = None  # set an int for smoke tests