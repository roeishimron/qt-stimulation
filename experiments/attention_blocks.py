from dataclasses import dataclass
from typing import List, Union
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from numpy import array, ones, pi, uint
from numpy.random import choice, randint, uniform

from animator import AppliableMixture, AppliableText
from experiments.constant_stimuli.dots_generator import (
    GenerationResult, GroupModification, GroupProperties, INCOHERENT,
    Incoherent, continue_moving_dots, generate_moving_dots,
)
from realtime_experiment import RealtimeViewingExperiment
from response_recorder import KeyRecorder
from soft_serial import SoftSerial
from stims import Dot as RenderDot, array_into_pixmap, fill_with_dots

from logging import getLogger
logger = getLogger(__name__)

# Timing primitives
SCREEN_REFRESH_RATE = 60
QUANTA_SEC = 2
TOTAL_SEC = 120
AMOUNT_OF_TRIALS = 3

# Derived
QUANTA_FRAMES = QUANTA_SEC * SCREEN_REFRESH_RATE
SUB_QUANTA_FRAMES = QUANTA_FRAMES // 2
TOTAL_QUANTAS = TOTAL_SEC // QUANTA_SEC
TOTAL_FRAMES = TOTAL_QUANTAS * QUANTA_FRAMES

# Block-length distribution: steady_state_quantas ~ U{0..4}
MIN_STEADY_STATE = 0
MAX_STEADY_STATE = 4

# Visual / motion
DOT_RADIUS = 20
AMOUNT_OF_DOTS = 50
VELOCITY = 8
GRID_COMPRESSION = DOT_RADIUS
COLORS = (-1, +1)

# SSVEP cycle times. LCM(6, 4) = 12; SUB_QUANTA_FRAMES (=60) % 12 == 0.
CYCLE_A = uint(6)
CYCLE_B = uint(4)

# Fixation cross
CROSS_FONT_SIZE = 50
CROSS_COLORS = {-1: Qt.GlobalColor.black, +1: Qt.GlobalColor.white}


@dataclass(frozen=True)
class Block:
    steady_state_quantas: int
    attention_color: int
    coherent_color: int
    coherent_direction: float


@dataclass(frozen=True)
class Segment:
    duration_frames: int
    modifications: List[GroupModification]
    fixation_color: int


def roll_blocks() -> List[Block]:
    """Sample blocks until they sum to exactly TOTAL_QUANTAS quantas."""
    blocks: List[Block] = []
    remaining = TOTAL_QUANTAS
    while remaining > 0:
        max_steady = min(MAX_STEADY_STATE, remaining - 1)
        steady = int(randint(MIN_STEADY_STATE, max_steady + 1))
        blocks.append(Block(
            steady_state_quantas=steady,
            attention_color=int(choice(COLORS)),
            coherent_color=int(choice(COLORS)),
            coherent_direction=float(uniform(0, 2 * pi)),
        ))
        remaining -= 1 + steady
    return blocks


def to_segments(blocks: List[Block]) -> List[Segment]:
    """Expand each block into 1-3 segments (steady-state, trial-1st-sec, trial-2nd-sec)."""
    segments: List[Segment] = []
    for block in blocks:
        if block.steady_state_quantas > 0:
            segments.append(Segment(
                duration_frames=block.steady_state_quantas * QUANTA_FRAMES,
                modifications=[
                    GroupModification(0.5, None, COLORS[0]),
                    GroupModification(0.5, None, COLORS[1]),
                ],
                fixation_color=block.attention_color,
            ))
        # Trial first second: coherent_color goes coherent in coherent_direction.
        segments.append(Segment(
            duration_frames=SUB_QUANTA_FRAMES,
            modifications=[
                GroupModification(
                    0.5,
                    block.coherent_direction if c == block.coherent_color else None,
                    c,
                )
                for c in COLORS
            ],
            fixation_color=block.attention_color,
        ))
        # Trial second second: coherent_color back to incoherent; other preserved.
        segments.append(Segment(
            duration_frames=SUB_QUANTA_FRAMES,
            modifications=[
                GroupModification(
                    0.5,
                    INCOHERENT if c == block.coherent_color else None,
                    c,
                )
                for c in COLORS
            ],
            fixation_color=block.attention_color,
        ))
    return segments


def initial_props(first_seg: Segment, max_lifetime: int) -> List[GroupProperties]:
    """Translate a segment's modifications into GroupProperties for the first call.
    GroupModification.direction None or INCOHERENT both map to GroupProperties.direction=None
    (incoherent at init); a float coherent direction is passed through."""
    cycle_times = (CYCLE_A, CYCLE_B)
    max_cycles = (max_lifetime // int(CYCLE_A), max_lifetime // int(CYCLE_B))
    out: List[GroupProperties] = []
    for mod, ct, mc in zip(first_seg.modifications, cycle_times, max_cycles):
        d: Union[float, None] = (
            float(mod.direction)
            if isinstance(mod.direction, (int, float)) and not isinstance(mod.direction, bool)
            else None
        )
        out.append(GroupProperties(
            ratio=mod.ratio, direction=d, cycle_time=ct,
            color=mod.color, max_cycles=mc,
        ))
    return out


def render(simulator_frame, fixation_color: int, size: int) -> AppliableMixture:
    moving = [
        RenderDot(int(d.r), array([d.x, d.y], dtype=int),
                  d.color * ones((2 * d.r, 2 * d.r)))
        for d in simulator_frame
    ]
    return AppliableMixture([
        array_into_pixmap(fill_with_dots(size, [], moving, 0, 0)),
        AppliableText("+", CROSS_FONT_SIZE, CROSS_COLORS[fixation_color]),
    ])


def generate_stimulus(blocks: List[Block], size: int):
    """Run the simulator end-to-end, returning (all_frames, fixation_per_frame)."""
    segments = to_segments(blocks)
    max_lifetime = size // VELOCITY // 2

    result: GenerationResult = generate_moving_dots(
        AMOUNT_OF_DOTS, DOT_RADIUS, size,
        segments[0].duration_frames, VELOCITY,
        initial_props(segments[0], max_lifetime),
        grid_compression=GRID_COMPRESSION,
    )
    all_frames = list(result.frames)
    fixation_per_frame = (
        [segments[0].fixation_color] * segments[0].duration_frames
    )
    for seg in segments[1:]:
        result = continue_moving_dots(result, seg.duration_frames, seg.modifications)
        all_frames.extend(result.frames)
        fixation_per_frame.extend([seg.fixation_color] * seg.duration_frames)
    return all_frames, fixation_per_frame


def run():
    app = QApplication(sys.argv)
    screen_height = app.primaryScreen().geometry().height()
    size = int(screen_height * 5 / 6)

    def trial_stimuli():
        blocks = roll_blocks()
        logger.info(f"rolled {len(blocks)} blocks covering "
                    f"{sum(1 + b.steady_state_quantas for b in blocks)} quantas")
        all_frames, fixation_per_frame = generate_stimulus(blocks, size)
        logger.info(f"generated {len(all_frames)} frames")
        return (
            (render(f, c, size), uint(1))
            for f, c in zip(all_frames, fixation_per_frame)
        )

    trials = (trial_stimuli() for _ in range(AMOUNT_OF_TRIALS))

    recorder = KeyRecorder()
    experiment = RealtimeViewingExperiment(
        trials,
        SoftSerial(),
        use_step=True,
        show_fixation_cross=False,
        on_trial_start=recorder.experiment_start,
        stimuli_on_keypress=recorder.record_key_response,
    )

    experiment.showFullScreen()
    app.exec()


if __name__ == "__main__":
    run()
