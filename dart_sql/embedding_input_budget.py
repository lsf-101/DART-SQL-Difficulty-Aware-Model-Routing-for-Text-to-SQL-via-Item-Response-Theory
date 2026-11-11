"""Shared, deterministic token budget for BIRD question/evidence/schema embeddings.

The training and inference pipelines must send the same representation to the
same embedding model. Only the schema suffix can be shortened; question and
evidence are kept intact. Token counting uses the exact newline normalization
performed immediately before the embedding API request.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


EMBEDDING_INPUT_POLICY_VERSION = "question_hint_schema_prefix_v1"
DEFAULT_EMBEDDING_TOKEN_BUDGET = 7800
EMBEDDING_API_MAX_INPUT_TOKENS = 8192
TRUNCATION_MARKER = " [SCHEMA_TRUNCATED]"


@dataclass(frozen=True)
class BudgetedEmbeddingText:
    text: str
    truncated: bool
    original_tokens: int | None
    final_tokens: int | None


def normalize_for_embedding_api(text: str) -> str:
    """Match the current training/test request path exactly."""
    return text.replace("\n", " ")


@lru_cache(maxsize=8)
def _encoding_for_model(model: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "Long embedding inputs require exact token counting with tiktoken. Run "
            "python -m pip install tiktoken in the environment used by 2PL.py and test_2PL.py"
        ) from exc
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError as exc:
        raise ValueError(
            f"tiktoken does not recognize embedding model {model!r}; check the model name and tokenizer"
        ) from exc


def prepare_embedding_text(
    question: str,
    evidence: str,
    schema: str,
    *,
    model: str = "text-embedding-3-large",
    token_budget: int = DEFAULT_EMBEDDING_TOKEN_BUDGET,
) -> BudgetedEmbeddingText:
    """Keep question/evidence and the longest fitting schema character prefix.

    The byte-length fast path is safe for the supported byte-pair tokenizer:
    each token contains at least one UTF-8 byte. Longer texts are measured
    with the model tokenizer, including the exact API newline normalization.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise ValueError("embedding_token_budget must be a positive integer")
    if not 0 < token_budget < EMBEDDING_API_MAX_INPUT_TOKENS:
        raise ValueError(
            f"embedding_token_budget must lie between 1 and "
            f"{EMBEDDING_API_MAX_INPUT_TOKENS - 1}"
        )

    prefix = f"Question: {question}\nHint: {evidence}\nDatabase Schema:\n"
    full_text = prefix + schema
    normalized = normalize_for_embedding_api(full_text)
    if len(normalized.encode("utf-8")) <= token_budget:
        return BudgetedEmbeddingText(full_text, False, None, None)

    encoding = _encoding_for_model(model)

    def count_tokens(text: str) -> int:
        return len(encoding.encode(normalize_for_embedding_api(text), disallowed_special=()))

    original_tokens = count_tokens(full_text)
    if original_tokens <= token_budget:
        return BudgetedEmbeddingText(full_text, False, original_tokens, original_tokens)

    if count_tokens(prefix + TRUNCATION_MARKER) > token_budget:
        raise ValueError(
            "Question and Hint, including fixed labels, already exceed the embedding token budget. "
            "Schema-only truncation cannot process this item."
        )

    # Cut by character, rather than decoding a partial token sequence, so
    # multibyte Chinese characters and identifiers are never corrupted.
    low, high = 0, len(schema)
    best = 0
    while low <= high:
        midpoint = (low + high) // 2
        candidate = prefix + schema[:midpoint] + TRUNCATION_MARKER
        if count_tokens(candidate) <= token_budget:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1

    budgeted = prefix + schema[:best] + TRUNCATION_MARKER
    final_tokens = count_tokens(budgeted)
    if final_tokens > token_budget:
        raise RuntimeError("Internal error: the truncated embedding input still exceeds the token budget")
    return BudgetedEmbeddingText(budgeted, True, original_tokens, final_tokens)
