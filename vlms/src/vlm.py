import base64
import os


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _mime(path):
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


class OpenAIVLM:
    """Works for OpenAI and any OpenAI-compatible endpoint (Qwen/DashScope, vLLM, Ollama...)."""

    def __init__(self, model, base_url=None, api_key=None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
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
            max_tokens=50,
            temperature=0,
        )
        return r.choices[0].message.content.strip()


class GeminiVLM:
    def __init__(self, model, api_key=None):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def ask(self, prompt, image_paths):
        from google.genai import types
        parts = [prompt]
        for p in image_paths:
            with open(p, "rb") as f:
                parts.append(types.Part.from_bytes(data=f.read(), mime_type=_mime(p)))
        r = self.client.models.generate_content(model=self.model, contents=parts)
        return r.text.strip()


class AnthropicVLM:
    def __init__(self, model, api_key=None):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

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
            max_tokens=50,
            messages=[{"role": "user", "content": content}],
        )
        return r.content[0].text.strip()


def get_vlm(provider, model, **kwargs):
    providers = {
        "openai": OpenAIVLM,
        "qwen": OpenAIVLM,      # pass base_url of the OpenAI-compatible endpoint
        "gemini": GeminiVLM,
        "anthropic": AnthropicVLM,
    }
    return providers[provider](model, **kwargs)
