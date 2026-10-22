"""Default embedding selector: preserve embeddings without selection."""
from torch import nn


class no_selection(nn.Identity):
    pass
