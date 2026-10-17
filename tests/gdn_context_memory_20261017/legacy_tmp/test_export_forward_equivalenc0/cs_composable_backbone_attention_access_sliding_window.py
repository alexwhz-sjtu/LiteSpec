# Generated inference bundle from composable/backbone/attention/access/sliding_window.py

"""W historical positions plus the WHOLE bidirectional current block."""
from .cs_composable_backbone_attention_access_base import build_visibility

def sliding_window_visibility(inputs, positions, valid, documents, query_length, window_size):
    return build_visibility(inputs, positions, valid, documents, query_length, window_size)
