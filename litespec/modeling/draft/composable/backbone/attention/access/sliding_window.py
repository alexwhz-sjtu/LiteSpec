"""W historical positions plus the WHOLE bidirectional current block."""

from .base import build_visibility


def sliding_window_visibility(
    inputs, positions, valid, documents, query_length, window_size
):
    return build_visibility(
        inputs, positions, valid, documents, query_length, window_size
    )
