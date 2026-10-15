from .dense import dense_visibility
from .sliding_window import sliding_window_visibility

ACCESS = {"dense": dense_visibility, "sliding_window": sliding_window_visibility}


def build_access(config):
    return ACCESS[config.get("access", "dense")]
