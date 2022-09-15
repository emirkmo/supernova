import enum


class StrEnum(str, enum.Enum):
    """Enum with string values."""
    def __str__(self) -> str:
        return str(self.value)
