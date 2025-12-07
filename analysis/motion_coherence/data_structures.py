from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional, Union, List

@dataclass
class SessionData:
    timestamp: int
    coherences: NDArray
    successes: NDArray
    directions: NDArray

@dataclass
class Fixed:
    session: SessionData

@dataclass
class Roving:
    session: SessionData

Experiment = Union[Fixed, Roving]

@dataclass
class GroupedData:
    coherences: NDArray
    successes: NDArray

@dataclass
class TrialGroups:
    same: GroupedData
    opposite: GroupedData
    deg90: GroupedData