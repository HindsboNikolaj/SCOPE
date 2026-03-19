# thinking.py — Thinking mode helpers for SLM planners

class ThinkingMode(str):
    """Thinking mode constants for different SLM architectures."""
    NEVER  = "never"
    TOGGLE = "toggle"
    ALWAYS = "always"
    LEVELS = "levels"

# Known model thinking behaviors.
# Keys use HuggingFace model paths. The lookup function does substring matching,
# so Ollama tags like "qwen3:4b" will match "Qwen/Qwen3-4B" automatically.
# Add your own models here if they support thinking toggles.
MODEL_CATALOG = {
    # Qwen3 family — all support thinking toggle
    "Qwen/Qwen3-4B":               {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-4B-FP8":           {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-8B":               {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-30B-A3B":          {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-30B-A3B-FP8":      {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-32B":              {"thinking": ThinkingMode.TOGGLE},
    "Qwen/Qwen3-Next-80B-A3B":     {"thinking": ThinkingMode.TOGGLE},
    # OpenAI reasoning models
    "openai/gpt-oss-20b":          {"thinking": ThinkingMode.LEVELS, "level_choices": ["low", "medium", "high"]},
    "openai/gpt-oss-120b":         {"thinking": ThinkingMode.LEVELS, "level_choices": ["low", "medium", "high"]},
}

def thinking_mode_for_model(model_id: str) -> str:
    """Determine the thinking mode for a given model ID."""
    if model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id]["thinking"]
    for k, v in MODEL_CATALOG.items():
        if k.lower() in model_id.lower():
            return v["thinking"]
    return ThinkingMode.NEVER
