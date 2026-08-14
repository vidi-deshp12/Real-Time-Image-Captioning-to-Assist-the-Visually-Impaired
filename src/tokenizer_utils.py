"""
Tokenizer setup — built once, reused everywhere.
Never call GPT2Tokenizer.from_pretrained("gpt2") directly outside this module;
always load from the saved checkpoint dir so train/inference vocab stays in sync.
"""

from transformers import GPT2Tokenizer


def build_tokenizer() -> GPT2Tokenizer:
    """Create tokenizer with special tokens. Use only during training setup."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "bos_token": "<start>",
        "eos_token": "<end>",
        "pad_token": "<pad>",
    })
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    return tokenizer


def load_tokenizer(tokenizer_dir: str) -> GPT2Tokenizer:
    """Load the tokenizer saved alongside a checkpoint. Use at inference time."""
    return GPT2Tokenizer.from_pretrained(tokenizer_dir)