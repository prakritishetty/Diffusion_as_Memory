"""HuggingFace loading helpers.

`tiny://...` builds a small T5 from a config instead of downloading weights. That is what lets the
whole P0 -> P1 -> P2 pipeline run as a CPU unit test with no network and no cached checkpoints,
which is the difference between "we think it will run on the GPU" and "we know the wiring is
correct".
"""

from __future__ import annotations

TINY_PREFIX = "tiny://"


def is_tiny(model_name: str | None) -> bool:
    return bool(model_name) and model_name.startswith(TINY_PREFIX)


def tiny_config(
    vocab_size: int = 128,
    d_model: int = 64,
    d_ff: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
):
    from transformers import T5Config

    return T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        d_kv=d_model // num_heads,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        relative_attention_num_buckets=8,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        is_encoder_decoder=True,
    )


def _tiny_kwargs(model_name: str) -> dict:
    """`tiny://d=64,vocab=128` -> {"d_model": 64, "vocab_size": 128}."""
    spec = model_name[len(TINY_PREFIX):]
    alias = {"d": "d_model", "vocab": "vocab_size", "ff": "d_ff",
             "layers": "num_layers", "heads": "num_heads"}
    out: dict = {}
    for part in filter(None, spec.split(",")):
        if "=" in part:
            k, v = part.split("=", 1)
            out[alias.get(k.strip(), k.strip())] = int(v)
    return out


def load_encoder(model_name: str):
    """Return (encoder_module, hidden_size)."""
    if is_tiny(model_name):
        from transformers import T5EncoderModel
        cfg = tiny_config(**_tiny_kwargs(model_name))
        return T5EncoderModel(cfg), cfg.d_model

    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_name)
    if getattr(cfg, "is_encoder_decoder", False):
        from transformers import T5EncoderModel
        try:
            enc = T5EncoderModel.from_pretrained(model_name)
        except Exception:
            enc = AutoModel.from_pretrained(model_name).encoder
    else:
        enc = AutoModel.from_pretrained(model_name)
    hidden = getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size", None)
    return enc, int(hidden)


def load_seq2seq(model_name: str):
    if is_tiny(model_name):
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration(tiny_config(**_tiny_kwargs(model_name)))

    from transformers import AutoModelForSeq2SeqLM
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)
