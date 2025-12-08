from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional, Union, List, Tuple, Iterable, Dict

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

class Subject:
    def __init__(self, sessions: Iterable[Experiment]):
        self.sessions: Tuple[Experiment, ...] = tuple(sorted(sessions, key=lambda s: s.session.timestamp))

    def is_fixed_first(self) -> bool:
        return isinstance(self.sessions[0], Fixed)

    def __eq__(self, other):
        if not isinstance(other, Subject):
            return NotImplemented
        return self.sessions == other.sessions

@dataclass
class GroupedData:
    coherences: NDArray
    successes: NDArray

@dataclass
class TrialGroups:
    same: GroupedData
    opposite: GroupedData
    deg90: GroupedData

@dataclass
class ExperimentalGroups:
    fixed_first: List[Experiment]
    fixed_second: List[Experiment]
    roving_first: List[Experiment]
    roving_second: List[Experiment]

@dataclass
class Trajectories:
    fixed_first: NDArray
    fixed_second: NDArray
    roving_first: NDArray
    roving_second: NDArray

    def to_dict(self) -> Dict[str, NDArray]:
        return {
            "Fixed First": self.fixed_first,
            "Fixed Second": self.fixed_second,
            "Roving First": self.roving_first,
            "Roving Second": self.roving_second
        }