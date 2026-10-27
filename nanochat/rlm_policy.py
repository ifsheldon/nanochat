"""Explicit training policies shared by the trainer and launch tools."""

from enum import Enum


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FA3 = "fa3"

    def __str__(self):
        return self.value


class InputEmbeddingFreeze(str, Enum):
    NONE = "none"
    SCHEDULED = "scheduled"
    PERMANENT = "permanent"

    def __str__(self):
        return self.value
