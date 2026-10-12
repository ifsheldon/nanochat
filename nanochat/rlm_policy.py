"""Input-table training policies shared by the trainer and launch tools."""

from enum import Enum


class InputEmbeddingFreeze(str, Enum):
    NONE = "none"
    SCHEDULED = "scheduled"
    PERMANENT = "permanent"

    def __str__(self):
        return self.value
