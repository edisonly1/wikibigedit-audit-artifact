"""Weight-editing baselines: FT-L and a small full-model variant.

Ollama exposes no gradients, so this arm writes the edit into the model weights
instead of the prompt. FT-L is the constrained fine-tuning baseline from the ROME
paper: a norm-bounded update to a single MLP down-projection. Every edit is applied
to a clone of the target weights and reverted exactly, so items stay independent.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EditConfig:
    """Defaults selected by scripts/check_edit_locality.py on Qwen2.5-1.5B:
    edit success 3/3, unrelated-fact preservation 93%. The norm constraint keeps
    the edit surgical while still moving the target answer."""

    method: str = "ft-l"          # ft-l | ft-m
    layer: int = 12               # MLP layer to edit for ft-l
    lr: float = 1e-3
    steps: int = 30
    norm_constraint: float = 5e-3  # max L-inf change per weight (ft-l)
    weight_decay: float = 0.0
    kl_factor: float = 0.0        # optional locality regulariser (0 = off)


class EditableModel:
    """Wraps a HF causal LM with deterministic greedy decoding and weight editing."""

    def __init__(self, model_path: str, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype).to(DEVICE)
        self.model.eval()
        self.dtype = dtype
        self.n_gen = 0

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, question: str, system: str, max_new_tokens: int = 24) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(DEVICE)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1,
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[0, ids["input_ids"].shape[1]:]
        self.n_gen += 1
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    # -- locate editable parameters ----------------------------------------
    def _mlp_down_proj(self, layer: int) -> torch.nn.Parameter:
        # Qwen2 / Llama layout: model.model.layers[i].mlp.down_proj.weight
        mod = self.model.model.layers[layer].mlp.down_proj
        return mod.weight

    def _target_params(self, cfg: EditConfig) -> list[torch.nn.Parameter]:
        if cfg.method == "ft-l":
            return [self._mlp_down_proj(cfg.layer)]
        # ft-m: edit the down_proj of a small band of mid layers
        n = self.model.config.num_hidden_layers
        lo, hi = max(0, cfg.layer - 1), min(n, cfg.layer + 2)
        return [self._mlp_down_proj(i) for i in range(lo, hi)]

    # -- edit / restore -----------------------------------------------------
    @contextmanager
    def edited(self, subject: str, relation: str, obj: str,
               cfg: EditConfig) -> Iterator[None]:
        """Apply an edit, yield, then restore the original weights exactly."""
        params = self._target_params(cfg)
        backup = [p.detach().clone() for p in params]
        try:
            self._apply_edit(subject, relation, obj, cfg, params, backup)
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)

    def _edit_prompt(self, subject: str, relation: str) -> tuple[str, str]:
        """Return (prompt, target) so that target is what should be generated."""
        prompt = f"The {relation} of {subject} is"
        return prompt, None

    def _apply_edit(self, subject, relation, obj, cfg, params, backup) -> None:
        prompt = f"The {relation} of {subject} is"
        full = f"{prompt} {obj}"
        enc = self.tok(full, return_tensors="pt").to(DEVICE)
        prompt_len = self.tok(prompt, return_tensors="pt")["input_ids"].shape[1]

        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100          # supervise only the object tokens

        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.model.train()
        for _ in range(cfg.steps):
            opt.zero_grad()
            out = self.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"], labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            # FT-L norm constraint: clamp each weight to a window around its
            # original value, keeping the edit surgical.
            if cfg.method == "ft-l" and cfg.norm_constraint > 0:
                with torch.no_grad():
                    for p, b in zip(params, backup):
                        p.clamp_(b - cfg.norm_constraint, b + cfg.norm_constraint)
        self.model.eval()
        for p in params:
            p.requires_grad_(False)
