import torch
import transformers


MODEL_MAP = {
    "qwen": "Qwen/Qwen2.5-7B",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}
CHAT_MODELS = frozenset({"phi"})


def model_load_kwargs_for_model(
    model_name, flash_attn_available=None, device_map="cuda:0"
):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")
    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": model_name == "phi"}
    if device_map and str(device_map).lower() != "none":
        kwargs["device_map"] = device_map
    if model_name == "phi":
        if flash_attn_available is None:
            flash_attn_available = transformers.utils.is_flash_attn_2_available()
        if not flash_attn_available:
            raise RuntimeError("model=phi requires FlashAttention 2 and a CUDA device.")
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def model_input_device(model):
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is not None and weight.device.type != "meta":
        return weight.device
    return next(model.parameters()).device


def load_tokenizer(model_name):
    return transformers.AutoTokenizer.from_pretrained(
        MODEL_MAP[model_name], trust_remote_code=model_name == "phi"
    )


def load_model(model_name, device="cuda", device_map="cuda:0"):
    if str(device).startswith("cpu"):
        device_map = "none"
    kwargs = model_load_kwargs_for_model(model_name, device_map=device_map)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name], **kwargs
    )
    if "device_map" not in kwargs:
        model = model.to(device)
    return model.eval()
