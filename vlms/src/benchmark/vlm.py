import base64
import json
import os


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime(path):
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def _clean(key):
    return key.strip() if key else key


class OpenAIVLM:
    """Works for OpenAI and any OpenAI-compatible endpoint (vLLM, HF router, DashScope...)."""

    def __init__(self, model, base_url=None, api_key=None, max_tokens=2000, **kwargs):
        from openai import OpenAI
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url,
            api_key=_clean(api_key or os.environ.get("OPENAI_API_KEY")) or "local",
        )

    def ask(self, prompt, image_paths):
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_mime(p)};base64,{_b64(p)}"},
            })
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=self.max_tokens,
        )
        return (r.choices[0].message.content or "").strip()


class GeminiVLM:
    def __init__(self, model, api_key=None, **kwargs):
        from google import genai
        self.model = model
        self.client = genai.Client(
            api_key=_clean(api_key or os.environ.get("GEMINI_API_KEY")))

    def ask(self, prompt, image_paths):
        from google.genai import types
        parts = [prompt]
        for p in image_paths:
            with open(p, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=_mime(p)))
        r = self.client.models.generate_content(model=self.model, contents=parts)
        return (r.text or "").strip()


class AnthropicVLM:
    def __init__(self, model, api_key=None, max_tokens=2000, **kwargs):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key=_clean(api_key or os.environ.get("ANTHROPIC_API_KEY")))

    def ask(self, prompt, image_paths):
        content = []
        for p in image_paths:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": _mime(p), "data": _b64(p)},
            })
        content.append({"type": "text", "text": prompt})
        r = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return next((b.text for b in r.content if b.type == "text"), "").strip()


class LocalVLM:
    """Runs a HF Qwen-VL-style model in-process, optionally with a LoRA/PEFT adapter.

    `model` is the base checkpoint (e.g. Qwen/Qwen3-VL-8B-Instruct) or "auto" to read
    base_model_name_or_path from the adapter's adapter_config.json.
    Not thread-safe: run with --concurrency 1.
    """

    def __init__(self, model, adapter=None, max_tokens=2000, dtype="bfloat16",
                 merge=True, **kwargs):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = model
        if adapter and (not model or model == "auto"):
            cfg = os.path.join(adapter, "adapter_config.json")
            base = json.load(open(cfg))["base_model_name_or_path"]
        self.max_tokens = max_tokens
        torch_dtype = getattr(torch, dtype)

        # processor: prefer the one saved next to the adapter (keeps any pixel-budget
        # settings used during training), fall back to the base model's
        proc_src = adapter if adapter and os.path.exists(
            os.path.join(adapter, "preprocessor_config.json")) else base
        self.processor = AutoProcessor.from_pretrained(proc_src)

        self.model = AutoModelForImageTextToText.from_pretrained(
            base, torch_dtype=torch_dtype, device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            if merge:
                self.model = self.model.merge_and_unload()
        self.model.eval()

    def ask(self, prompt, image_paths):
        import torch
        from PIL import Image

        content = [{"type": "image"} for _ in image_paths] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                      do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def get_vlm(provider, model, **kwargs):
    providers = {
        "openai": OpenAIVLM,
        "qwen": OpenAIVLM,
        "gemini": GeminiVLM,
        "anthropic": AnthropicVLM,
        "local": LocalVLM,
    }
    return providers[provider](model, **kwargs)