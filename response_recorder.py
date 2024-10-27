from dataclasses import dataclass
from typing import List

@dataclass
class Stimuli:
    responded: bool
    showed_at: int


class ResponseRecorder:
    stimuli_times: List[Stimuli]
    resopnse_delays: List[int]

    def __init__(self):
        self.stimuli_times = [Stimuli(True, 0)]
        self.resopnse_delays = []

    def record_stimuli_show(self):
        self.stimuli_times.append(Stimuli(False, time_ns()))

    def record_response(self):
        if self.stimuli_times[-1].responded:
            return

        diff = time_ns() - self.stimuli_times[-1].showed_at
        self.stimuli_times[-1].responded = True
        self.resopnse_delays.append(diff)
        print(f"got response after {diff} ns (or {diff/10**6} ms)")

