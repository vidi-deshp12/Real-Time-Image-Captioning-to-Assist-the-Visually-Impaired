"""
ClipCap model definition.
MLP maps CLIP embeddings → GPT-2 prefix tokens.
ClipCaptionModel wires CLIP prefix + GPT-2 together.
ClipCaptionPrefix freezes GPT-2 and only trains the projection.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

T = torch.Tensor


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias: bool = True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: T) -> T:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512, vocab_size: Optional[int] = None):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")

        # Resize embeddings to include <start>/<end>/<pad>.
        # Must happen before training so special tokens get real trained embeddings.
        if vocab_size is not None:
            self.gpt.resize_token_embeddings(vocab_size)

        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if prefix_length > 10:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            ))

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            # -100 so CrossEntropyLoss ignores prefix positions.
            # Original bug: used zeros (token '!') which corrupted the loss.
            dummy_labels = torch.full(
                (tokens.shape[0], self.prefix_length),
                fill_value=-100,
                dtype=torch.long,
                device=tokens.device,
            )
            labels = torch.cat((dummy_labels, tokens), dim=1)

        return self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)


class ClipCaptionPrefix(ClipCaptionModel):
    """Only trains the CLIP→GPT-2 projection; GPT-2 weights stay frozen."""

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt.eval()
        return self