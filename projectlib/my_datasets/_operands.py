
from typing import Callable, Dict, Literal

type Operation = Literal["+", "-"]
OPERATION: Dict[Operation, Callable[[float, float], float]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
}