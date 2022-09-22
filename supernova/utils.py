import enum
from pathlib import Path
import astropy.units as u
Number = int | float | u.Quantity
PathType = Path | str


class StrEnum(str, enum.Enum):
    """Enum with string values."""
    def __str__(self) -> str:
        return str(self.value)
