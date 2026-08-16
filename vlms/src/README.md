# VLM CAD Benchmark

## Files

- `prompts.py` — all prompts in one place
- `vlm.py` — provider clients (OpenAI, Qwen/any OpenAI-compatible endpoint, Gemini, Anthropic)
- `tasks.py` — item builders for tasks 1–7
- `metrics.py` — accuracy, macro precision/recall/F1, confusion matrix, numeric metrics
- `run.py` — CLI entry point

## Install

```bash
pip install openai google-genai anthropic scikit-learn numpy
```

Set the API key for the provider you use:
`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`.

## Dataset manifest

One JSON file listing every model (see docstring in `tasks.py`):

```json
[
  {
    "id": "model_0001",
    "complexity": "simple",
    "views": {
      "front": "data/model_0001/front.png",
      "back": "data/model_0001/back.png",
      "left": "data/model_0001/left.png",
      "right": "data/model_0001/right.png",
      "top": "data/model_0001/top.png",
      "bottom": "data/model_0001/bottom.png"
    },
    "drawing_dim": "data/model_0001/drawing_dim.png",
    "drawing_nodim": "data/model_0001/drawing_nodim.png",
    "symmetric": false,
    "dimensions": [{"question": "What is the overall length of the part?", "value": 120.0}],
    "counts": [{"question": "How many through-holes are visible?", "value": 4, "view": "front"}]
  }
]
```

`symmetric`, `dimensions`, and `counts` are optional. Models marked
`"symmetric": true` are excluded from tasks 2 and 5. Tasks 6 and 7 only use
models that have `dimensions` / `counts`.

## Run

```bash
# Task 1, GPT-4o, dimensioned drawings
python run.py --task 1 --manifest manifest.json --provider openai --model gpt-4o --dimensioned

# Task 2, Gemini
python run.py --task 2 --manifest manifest.json --provider gemini --model gemini-2.0-flash

# Task 3 with 10 candidates, Qwen via DashScope (OpenAI-compatible)
python run.py --task 3 --manifest manifest.json --provider qwen --model qwen-vl-max \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --n-candidates 10

# Task 5, Claude
python run.py --task 5 --manifest manifest.json --provider anthropic --model claude-sonnet-4-6

# Task 6, fewer items
python run.py --task 6 --manifest manifest.json --provider openai --model gpt-4o --n-items 200
```

Output: one JSON per run in `results/` containing config, per-item results
(raw response + parsed prediction), and metrics overall, by complexity class,
and (task 1) by negative tier.

## Adding a new provider

Add a class in `vlm.py` with an `ask(prompt, image_paths) -> str` method and
register it in `get_vlm`.
