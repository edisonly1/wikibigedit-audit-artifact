"""Deterministic Ollama client for the locality experiment.

We measure whether an answer changes after an edit, so decoding has to be
deterministic (temperature 0, top_k 1, fixed seed); `check_determinism` gates that.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

OLLAMA = "http://localhost:11434"

SYSTEM = (
    "You answer factual questions with a short noun phrase and nothing else.\n"
    "Rules:\n"
    "- Output only the answer itself, at most a few words.\n"
    "- Never write a sentence, an explanation, or a preamble.\n"
    "- Never write 'I think', 'my guess is', 'unknown', or 'I don't know'.\n"
    "- If you are unsure, still output your single best specific answer.\n"
    "Example question: What is the country of citizenship of Marie Curie?\n"
    "Example answer: Poland"
)


@dataclass
class GenResult:
    text: str
    ms: float


class Ollama:
    def __init__(self, model: str, seed: int = 0, num_predict: int = 24,
                 timeout: int = 180):
        self.model = model
        self.seed = seed
        self.num_predict = num_predict
        self.timeout = timeout
        self.session = requests.Session()
        self.n_calls = 0

    def chat(self, user: str, system: str = SYSTEM) -> GenResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": self.seed,
                "num_predict": self.num_predict,
                "repeat_penalty": 1.0,
            },
        }
        t0 = time.time()
        for attempt in range(4):
            try:
                r = self.session.post(f"{OLLAMA}/api/chat", json=payload,
                                      timeout=self.timeout)
                if r.status_code == 200:
                    txt = (r.json().get("message") or {}).get("content", "")
                    return GenResult(txt.strip(), (time.time() - t0) * 1000)
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"ollama call failed for model {self.model}")

    def generate(self, user: str, system: str = SYSTEM) -> str:
        self.n_calls += 1
        return self.chat(user, system).text


# ---------------------------------------------------------------------------
# answer normalisation
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"^(the|a|an)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s]", re.U)


def normalise(ans: str) -> str:
    """Canonical form for answer comparison.

    Deliberately conservative: strips punctuation, collapses whitespace, drops a
    leading article and lowercases. It does not stem or synonym-match, so two
    answers judged equal really are the same surface string.
    """
    if not isinstance(ans, str):
        return ""
    a = ans.strip().split("\n")[0]
    a = _PUNCT.sub(" ", a)
    a = re.sub(r"\s+", " ", a).strip().lower()
    a = _ARTICLES.sub("", a)
    return a.strip()


def same_answer(a: str, b: str) -> bool:
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return na == nb
    return na == nb


def contains_answer(haystack: str, needle: str) -> bool:
    """Whether a target answer appears in a response, token-boundary aware."""
    nh, nn = normalise(haystack), normalise(needle)
    if not nh or not nn:
        return False
    return re.search(rf"(?<!\w){re.escape(nn)}(?!\w)", nh) is not None


# ---------------------------------------------------------------------------

def check_determinism(model: str, prompts: list[str], repeats: int = 3,
                      warmup: bool = True) -> dict:
    """Run each prompt `repeats` times; report whether outputs are identical.

    A single mismatch invalidates the locality metric, so this is a gate, not a
    diagnostic.

    The first call after a model is loaded can differ from later ones (weights are
    still being paged onto the GPU, so the layer split differs). `warmup` issues a
    throwaway call first; the experiment runner must do the same.
    """
    cli = Ollama(model)
    mismatches = []
    lat = []
    if warmup:
        for _ in range(2):
            cli.chat("What is the capital of France?")
    for p in prompts:
        outs = []
        for _ in range(repeats):
            g = cli.chat(p)
            outs.append(g.text)
            lat.append(g.ms)
        if len(set(outs)) > 1:
            mismatches.append({"prompt": p, "outputs": outs})
    return {
        "model": model,
        "n_prompts": len(prompts),
        "repeats": repeats,
        "n_mismatched": len(mismatches),
        "deterministic": not mismatches,
        "mismatches": mismatches[:5],
        "mean_ms": sum(lat) / max(len(lat), 1),
    }
