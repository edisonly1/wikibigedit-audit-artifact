"""ROME rank-one editing (Meng et al., 2022).

Writes one fact into an MLP down-projection W as a rank-one update that maps the
subject key k* to an optimised value v*, kept minimal under the key covariance C:

    dW = (v* - W k*)(C^{-1} k*)^T / (k*^T C^{-1} k*).

Then W'k* = v* exactly and no gradient touches W. Edits clone W and revert it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .param_edit import EditableModel, DEVICE


@dataclass
class RomeConfig:
    layer: int = 12
    v_lr: float = 0.5
    v_steps: int = 40
    v_weight_decay: float = 1e-3
    kl_factor: float = 0.0
    cov_lambda: float = 0.05      # ridge on C, as a fraction of mean(diag)
    clamp_norm_factor: float = 8.0  # cap |v* - baseline| to this x baseline norm


def _subject_last_token(tok, prompt: str, subject: str) -> int:
    """Index (in the tokenisation of `prompt`) of the subject's last token.

    Falls back to the last non-special token of the prompt if the subject cannot
    be located, which keeps the edit well-defined for odd subject strings.
    """
    ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    # find the subject span by decoding growing suffixes -- robust to tokenizer
    sub_ids = tok(subject, add_special_tokens=False)["input_ids"]
    if sub_ids:
        n = len(sub_ids)
        for start in range(len(ids) - n, -1, -1):
            if ids[start:start + n].tolist() == sub_ids:
                return start + n - 1
    # fallback: last token before any trailing whitespace token
    return len(ids) - 1


class RomeEditor:
    """Adds ROME editing to an EditableModel. Covariance C is cached per layer."""

    def __init__(self, em: EditableModel, cfg: RomeConfig,
                 cov_path: str | None = None):
        self.em = em
        self.cfg = cfg
        self.model = em.model
        self.tok = em.tok
        self.down = self.model.model.layers[cfg.layer].mlp.down_proj  # W
        self.d_model, self.d_ffn = self.down.weight.shape
        self.cov_path = cov_path or f"data/rome_cov_layer{cfg.layer}.pt"
        self._Cinv = None

    # -- key covariance -----------------------------------------------------
    def estimate_cov(self, texts: list[str], max_positions: int = 40000) -> None:
        """Accumulate C = E[k k^T] over down_proj inputs at the edit layer."""
        if os.path.exists(self.cov_path):
            self._load_cov()
            return
        acc = torch.zeros(self.d_ffn, self.d_ffn, dtype=torch.float64, device=DEVICE)
        count = 0
        captured = {}

        def hook(_m, inp, _out):
            captured["k"] = inp[0].detach()

        h = self.down.register_forward_hook(hook)
        self.model.eval()
        with torch.no_grad():
            for t in texts:
                if count >= max_positions:
                    break
                ids = self.tok(t, return_tensors="pt", truncation=True,
                               max_length=64).to(DEVICE)
                if ids["input_ids"].shape[1] == 0:
                    continue
                self.model(**ids)
                k = captured["k"][0].to(torch.float64)      # [seq, d_ffn]
                acc += k.t() @ k
                count += k.shape[0]
        h.remove()
        C = (acc / max(count, 1)).to(torch.float32)
        os.makedirs(os.path.dirname(self.cov_path) or ".", exist_ok=True)
        torch.save({"C": C.cpu(), "count": count}, self.cov_path)
        print(f"[rome] estimated C from {count} key positions -> {self.cov_path}")
        self._finalise_cov(C)

    def _load_cov(self) -> None:
        d = torch.load(self.cov_path, map_location=DEVICE)
        self._finalise_cov(d["C"].to(DEVICE))
        print(f"[rome] loaded C ({d['count']} positions) from {self.cov_path}")

    def _finalise_cov(self, C: torch.Tensor) -> None:
        lam = self.cfg.cov_lambda * C.diagonal().mean()
        C = C + lam * torch.eye(self.d_ffn, device=DEVICE)
        self._Cinv = torch.linalg.inv(C.to(torch.float32))

    def _chat_prompt(self, subject: str, relation: str, system: str) -> str:
        q = f"What is the {relation} of {subject}?"
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": q}]
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)

    # -- value optimisation -------------------------------------------------
    def _compute_kv(self, subject: str, relation: str, target: str,
                    system: str = ("You answer factual questions with a short noun "
                                   "phrase and nothing else. Output only the "
                                   "answer, at most a few words.")):
        # Optimise in the SAME chat context the experiment generates in, so the
        # subject key at edit time matches the subject key at test time; otherwise
        # the rank-one edit (which fires on k*) does not transfer.
        prompt = self._chat_prompt(subject, relation, system)
        full = prompt + target
        pos = _subject_last_token(self.tok, prompt, subject)

        enc = self.tok(full, return_tensors="pt").to(DEVICE)
        prompt_len = self.tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100

        # capture baseline key and value at the subject position
        cap = {}

        def cap_hook(_m, inp, out):
            cap["k"] = inp[0][0, pos].detach().clone()
            cap["v0"] = out[0, pos].detach().clone()

        h = self.down.register_forward_hook(cap_hook)
        with torch.no_grad():
            self.model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"])
        h.remove()
        k_star = cap["k"].to(torch.float32)
        v0 = cap["v0"].to(torch.float32)

        # optimise a delta added to the down_proj output at `pos`
        delta = torch.zeros_like(v0, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.cfg.v_lr,
                               weight_decay=self.cfg.v_weight_decay)

        def add_hook(_m, _inp, out):
            out = out.clone()
            out[0, pos] = out[0, pos] + delta
            return out

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        for _ in range(self.cfg.v_steps):
            opt.zero_grad()
            hh = self.down.register_forward_hook(add_hook)
            out = self.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"], labels=labels)
            hh.remove()
            loss = out.loss
            loss.backward()
            opt.step()
            # clamp the value change to a norm ball around the baseline
            with torch.no_grad():
                maxn = self.cfg.clamp_norm_factor * v0.norm()
                if delta.norm() > maxn:
                    delta.mul_(maxn / delta.norm())
        v_star = (v0 + delta.detach()).to(torch.float32)
        return k_star, v_star

    # -- the rank-one edit --------------------------------------------------
    def edited(self, subject: str, relation: str, obj: str):
        return _RomeEditCtx(self, subject, relation, obj)

    def _apply(self, subject, relation, obj):
        if self._Cinv is None:
            raise RuntimeError("call estimate_cov() before editing")
        k_star, v_star = self._compute_kv(subject, relation, obj)
        W = self.down.weight.data                       # [d_model, d_ffn]
        Cinv_k = self._Cinv @ k_star                    # [d_ffn]
        denom = torch.dot(k_star, Cinv_k)
        residual = v_star - W @ k_star                  # [d_model]
        dW = torch.outer(residual, Cinv_k) / denom      # [d_model, d_ffn]
        W += dW.to(W.dtype)


class _RomeEditCtx:
    def __init__(self, editor: RomeEditor, subject, relation, obj):
        self.e = editor
        self.args = (subject, relation, obj)
        self._backup = None

    def __enter__(self):
        self._backup = self.e.down.weight.data.detach().clone()
        self.e._apply(*self.args)
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            self.e.down.weight.data.copy_(self._backup)
        return False
