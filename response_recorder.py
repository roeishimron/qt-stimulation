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
    available_stimuli: List[Stimuli]
    false_pressing: int

    MAXIMUM_RESPONSE_DELAY = 2 * 10**9

    def __init__(self):
        self.stimuli_times = [Stimuli(True, 0)]
        self.resopnse_delays = []
        self.false_pressing = 0
        self.available_stimuli = []

    def record_stimuli_show(self):
        stim = Stimuli(False, time_ns())
        self.stimuli_times.append(stim)
        self.available_stimuli.append(stim)

    def get_oldest_available(self, current_time: int) -> Stimuli | None:
        while len(self.available_stimuli) > 0:
            stim = self.available_stimuli.pop(0)
            if current_time - stim.showed_at < self.MAXIMUM_RESPONSE_DELAY and not stim.responded:
                return stim
        return None

    def record_response(self):
        now = time_ns()
 
        stim_responded = self.get_oldest_available(now)

        # the update may result an unavailable
        if stim_responded == None:
            self.false_pressing += 1
            return

        diff = now - stim_responded.showed_at
        stim_responded.responded = True
        self.resopnse_delays.append(diff)
        print(f"got response after {diff/10**6} ms")

    # adding the false-pressing to the amount of stimuli (so it counts like an unreported stimulus)
    def success_rate(self):
        if len(self.stimuli_times) == 1:
            # nothing with nothing is 1, otherwise 0
            return float(self.false_pressing == 0)

        return (len([w for w in self.resopnse_delays if w < self.MAXIMUM_RESPONSE_DELAY])
                / (len(self.stimuli_times) - 1 + self.false_pressing))
