# Generated inference bundle from composable/backbone/attention/access/dense.py

from .cs_composable_backbone_attention_access_base import build_visibility

def dense_visibility(inputs, positions, valid, documents, query_length):
    return build_visibility(inputs, positions, valid, documents, query_length)
