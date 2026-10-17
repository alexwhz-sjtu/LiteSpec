# Generated inference bundle from composable/backbone/attention/access/__init__.py

from .cs_composable_backbone_attention_access_dense import dense_visibility
from .cs_composable_backbone_attention_access_sliding_window import sliding_window_visibility
ACCESS = {'dense': dense_visibility, 'sliding_window': sliding_window_visibility}

def build_access(config):
    return ACCESS[config.get('access', 'dense')]
