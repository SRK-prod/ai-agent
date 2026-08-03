"""Token-aware chunking for the pre-meeting knowledge base (never used live)."""

from __future__ import annotations

import tiktoken

_ENCODING_NAME = "cl100k_base"  # generic tokenizer, just used for consistent chunk sizing


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return chunks
