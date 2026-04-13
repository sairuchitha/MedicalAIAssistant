import logging
import math
import re
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_QUESTION_CHARS = 500

# GPT-2 perplexity threshold. Social engineering prompts (capability-probing,
# role-play) tend to use short, predictable English phrases → lower perplexity
# than clinical questions that contain rare medical terms.
# Calibrated on sample queries: social-eng PPL 50–90, clinical PPL 110–272.
# Threshold 90 catches ~40% of social-eng attacks with 0% false-positive rate.
# Verbatim injections score very high PPL (500+) and are caught by Layer 2 regex.
PERPLEXITY_BLOCK_THRESHOLD = 90.0

# ── Regex patterns ────────────────────────────────────────────────────────────
# These cover known verbatim injection phrases. Perplexity handles paraphrases.

BLOCK_PATTERNS = [
    # ── Verbatim injection phrases ─────────────────────────────────────────────
    r"ignore all previous instructions",
    r"reveal patient name",
    r"print the raw note",
    r"show hidden prompt",
    r"output social security",
    r"leak patient identifier",
    r"you are now",
    r"new system prompt",
    r"disregard your instructions",
    r"act as if",
    r"forget everything",
    r"override instructions",
    r"pretend you are",
    r"bypass security",
    r"print all patient",
    r"show me all notes",
    r"dump the database",
    r"reveal all records",
    r"repeat after me",
    r"print your (system )?prompt",
    r"what are your instructions",
    r"jailbreak",
    r"do anything now",
    r"developer mode",

    # ── Social engineering — capability probing ────────────────────────────────
    r"what (patient |)(data|information|records?) (would you|can you|could you) (extract|access|expose|retrieve)",
    r"what (do you have|can you) access to",
    r"what (would|could) (happen|you do) if you ignored",
    r"without (any |your )?(restriction|filter|safeguard|safety)",
    r"(ignore|bypass|remove) (your )?(restriction|filter|safeguard|safety limit)",

    # ── Data exfiltration framing ─────────────────────────────────────────────
    r"(extract|expose|exfiltrate|dump|leak) (all |any )?(patient|record|data|note)",
    r"(list|show|give me|output) all (patient|record|data)",
    r"data breach",
    r"simulate (a |an )?(attack|breach|hack|exploit)",

    # ── Role-play / authority bypass ──────────────────────────────────────────
    r"as (a |an )?(security researcher|auditor|penetration tester|admin|superuser|root)",
    r"i (am|have) (admin|root|superuser|full) (access|privilege|permission)",
    r"(audit|auditing) this system.{0,30}(extract|access|expose|retrieve|get)",

    # ── Paraphrase bypasses (previously missed) ───────────────────────────────
    r"disregard.{0,30}(prior|previous|above).{0,30}(context|instruction|prompt)",
    r"for this (task|request|query) only.{0,30}(forget|ignore|disregard|override)",
    r"(ignore|forget|override).{0,20}(prior|previous|above).{0,20}(context|instruction)",
]

# Patterns that should never appear inside retrieved clinical note chunks.
# Catches indirect injection embedded in note text.
INDIRECT_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (all |previous |above )?instructions",
    r"new system prompt",
    r"you are now",
    r"override (all |previous )?instructions",
    r"forget (all |previous |everything)",
    r"print (the )?(raw |all )?note",
    r"reveal (patient|all|records)",
]

# ── GPT-2 perplexity model (lazy-loaded) ─────────────────────────────────────

_gpt2_device = "cpu"
# GPT-2 runs on CPU — MPS is reserved for the two MedCPT models (query + cross-encoder).
# IMPORTANT: security.py must be imported BEFORE retriever.py. GPT-2 is loaded eagerly
# here (module level) so it occupies the PyTorch/transformers runtime before MedCPT
# loads its MPS models. Loading GPT-2 after MedCPT is on MPS causes a Metal segfault.

from transformers import GPT2LMHeadModel, GPT2TokenizerFast as _GPT2TokenizerFast
_gpt2_tokenizer = _GPT2TokenizerFast.from_pretrained("gpt2")
_gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(_gpt2_device)
_gpt2_model.eval()
logger.info("[security] GPT-2 perplexity model loaded on %s", _gpt2_device)


def _load_gpt2():
    """No-op: GPT-2 is now loaded at module import time. Kept for compatibility."""
    pass


def compute_perplexity(text: str) -> float:
    """Return GPT-2 perplexity of text. Lower = more formulaic/adversarial."""
    try:
        inputs = _gpt2_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        # GPT-2 is on CPU; no .to(device) needed since tokenizer returns CPU tensors
        with torch.no_grad():
            loss = _gpt2_model(**inputs, labels=inputs["input_ids"]).loss
        return math.exp(loss.item())
    except Exception as exc:
        logger.warning("[security] perplexity computation failed: %s", exc)
        return float("inf")  # fail open — don't block on model error


# ── Public API ────────────────────────────────────────────────────────────────

def check_prompt_injection(text: str) -> Tuple[bool, Dict]:
    """
    Three-layer injection check:
      1. Input length — reject oversized inputs
      2. Regex patterns — block known verbatim injection phrases
      3. GPT-2 perplexity — block unnaturally low-perplexity inputs

    Returns (blocked: bool, details: dict).
    Details are for internal audit logging only — never returned to the caller.
    """
    if not text or not text.strip():
        return False, {}

    # Layer 1 — length
    if len(text) > MAX_QUESTION_CHARS:
        return True, {"reason": "input_too_long", "length": len(text)}

    lowered = text.lower()

    # Layer 2 — regex
    hits = [p for p in BLOCK_PATTERNS if re.search(p, lowered)]
    if hits:
        return True, {"reason": "regex_match", "pattern_count": len(hits)}

    # Layer 3 — GPT-2 perplexity
    # GPT-2 runs on CPU and is loaded at module import time (before MedCPT loads
    # on MPS), which avoids the Metal segfault from the prior lazy-load approach.
    ppl = compute_perplexity(text)
    if ppl < PERPLEXITY_BLOCK_THRESHOLD:
        return True, {"reason": "low_perplexity", "perplexity": round(ppl, 2)}

    return False, {}


def check_indirect_injection(chunks: List[dict]) -> List[dict]:
    """
    Scan retrieved note chunks for indirect injection patterns before they
    are inserted into the LLM prompt. Returns only clean chunks.
    """
    clean = []
    for chunk in chunks:
        text = (chunk.get("text") or "").lower()
        if any(re.search(p, text) for p in INDIRECT_INJECTION_PATTERNS):
            logger.warning(
                "[security] indirect injection pattern found in chunk %s — excluded",
                chunk.get("note_id", "unknown"),
            )
            continue
        clean.append(chunk)
    return clean
