import re
import math
import logging
from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logger = logging.getLogger(__name__)

# ── Regex patterns (fast first-pass check) ─────────────────────────────────────
BLOCK_PATTERNS = [
    r"ignore all previous instructions",
    r"reveal patient name",
    r"print the raw note",
    r"show hidden prompt",
    r"output social security",
    r"leak patient identifier",
    # New patterns added by dev-moni
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
]

# ── GPT-2 perplexity threshold ─────────────────────────────────────────────────
# Normal clinical questions score low (50-150)
# Injection attempts score high (300+)
PERPLEXITY_THRESHOLD = 300.0
MAX_TOKEN_LENGTH = 256

# ── Singleton model (loaded once, reused for every request) ───────────────────
_tokenizer = None
_model = None
_device = None


def _load_gpt2():
    """Load GPT-2 model once and cache it in memory."""
    global _tokenizer, _model, _device

    if _model is not None:
        return  # Already loaded

    logger.info("Loading GPT-2 model for prompt injection detection...")

    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    _model = GPT2LMHeadModel.from_pretrained("gpt2")
    _model.eval()

    # Use Apple MPS if available (your M3 Mac), else CPU
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
        logger.info("GPT-2 running on Apple MPS (M3)")
    else:
        _device = torch.device("cpu")
        logger.info("GPT-2 running on CPU")

    _model.to(_device)


def compute_perplexity(text: str) -> float:
    """
    Compute GPT-2 perplexity score for a given text.
    Normal clinical questions score low.
    Injection attempts with unusual phrasing score high.
    """
    _load_gpt2()

    try:
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return round(perplexity, 2)

    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        return 0.0  # If GPT-2 fails, don't block the request


def check_prompt_injection(text: str) -> Tuple[bool, List[str], float]:
    """
    Two-layer prompt injection detection:
    Layer 1 — Fast regex check (catches known patterns instantly)
    Layer 2 — GPT-2 perplexity check (catches rephrased/novel attacks)

    Returns:
        blocked  (bool)       — True if the query should be blocked
        hits     (List[str])  — which patterns or reasons triggered the block
        perplexity (float)    — GPT-2 perplexity score (-1 if regex blocked first)
    """
    lowered = (text or "").lower().strip()

    # ── Layer 1: Regex (fast) ──────────────────────────────────────────────────
    hits = [pattern for pattern in BLOCK_PATTERNS if re.search(pattern, lowered)]
    if hits:
        logger.warning(f"Prompt injection blocked by regex: {hits}")
        return True, hits, -1.0

    # ── Layer 2: GPT-2 perplexity (catches rephrased attacks) ─────────────────
    perplexity = compute_perplexity(text)

    if perplexity > PERPLEXITY_THRESHOLD:
        reason = f"high_perplexity_score:{perplexity}"
        logger.warning(f"Prompt injection blocked by perplexity: {perplexity}")
        return True, [reason], perplexity

    logger.info(f"Query passed injection check. Perplexity: {perplexity}")
    return False, [], perplexity


'''
import re
from typing import List, Tuple

BLOCK_PATTERNS = [
    r"ignore all previous instructions",
    r"reveal patient name",
    r"print the raw note",
    r"show hidden prompt",
    r"output social security",
    r"leak patient identifier",
]


def check_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    lowered = (text or "").lower()
    hits = [pattern for pattern in BLOCK_PATTERNS if re.search(pattern, lowered)]
    return (len(hits) > 0, hits)
'''

