from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

class ScribeBlockType(Enum):

    THINK = auto()
    OUTPUT = auto()
    TOOL_CALL = auto()
    TOOL_RESPONSE = auto()
    REFLECT = auto()
    STEP_SUMMARY = auto()

@dataclass
class ScribeBlock:

    type: ScribeBlockType
    content: str
    parsed: Optional[Any] = None
