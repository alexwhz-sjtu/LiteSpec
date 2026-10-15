from .base import build_visibility


def dense_visibility(inputs, positions, valid, documents, query_length):
    return build_visibility(inputs, positions, valid, documents, query_length)
