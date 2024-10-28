from dataclasses import dataclass
from typing import List
from time import time_ns

@dataclass
class Stimuli:
    responded: bool
    showed_at: int


class ResponseRecorder:
    stimuli_times: List[Stimuli]
    resopnse_delays: List[int]
    false_pressing: int

    def __init__(self):
        self.stimuli_times = [Stimuli(True, 0)]
        self.resopnse_delays = []
        self.false_pressing = 0

    def record_stimuli_show(self):
        self.stimuli_times.append(Stimuli(False, time_ns()))

    def record_response(self):
        if self.stimuli_times[-1].responded:
            self.false_pressing += 1
            return

        diff = time_ns() - self.stimuli_times[-1].showed_at
        self.stimuli_times[-1].responded = True
        self.resopnse_delays.append(diff)
        print(f"got response after {diff} ns (or {diff/10**6} ms)")

    # adding the false-pressing to the amount of stimuli (so it counts like an unreported stimulus)
    def success_rate(self):
        if len(self.stimuli_times) == 1:
            return int(self.false_pressing == 0) # nothing with nothing is 1, otherwise 0
        
        return len([w for w in self.resopnse_delays if w < 2 * 10**9]) / (len(self.stimuli_times) - 1 + self.false_pressing)
