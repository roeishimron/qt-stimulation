import random
import sys
from itertools import cycle
from typing import List
import numpy as np
from PySide6.QtWidgets import QApplication
from numpy.typing import NDArray
from stims import fill_with_dots, Dot, array_into_pixmap
from animator import OddballStimuli
from realtime_experiment import RealtimeViewingExperiment
from soft_serial import SoftSerial
from typing import Tuple, Optional
import math
from PySide6.QtGui import QGuiApplication


DIRECTIONS = [90]
FREQUENCIES = (0,)  # frequencies for each direction
NUMBER_OF_DOTS = 35
FRAME_SIZE = 1000
FRAME_RATE = 60
EXPERIMENT_TIME = 300  # in frames
COHERENCE = 0.2   # amount of dots moving in the coherent direction
LIFETIME_FRAMES: Optional[Tuple[int, int] | int] = (5,10)
LIFE_TIME_AVERAGE = sum(LIFETIME_FRAMES)/2

PATCH_VISIBLE = {}
PATCH_HIDDEN = {}
CENTER = np.array([FRAME_SIZE // 2, FRAME_SIZE // 2])
DEFAULT_VELOCITY = 12

def dist_to_angle(screen_size: float, distance: float) -> float:
    angle = math.atan2( (screen_size / 2.0) , distance)
    angle_deg = math.degrees(2*angle)
    return angle_deg

SCREEN_SIZE_IN_DEG = dist_to_angle(34, 40)
SCREEN_SIZE_IN_PIXELS = 1980
PIXELS_PER_DEG = SCREEN_SIZE_IN_PIXELS / SCREEN_SIZE_IN_DEG


PIXELS_PER_DEG = SCREEN_SIZE_IN_PIXELS / SCREEN_SIZE_IN_DEG
DOT_RADIUS_DEG = 0.5
RADIUS = int(DOT_RADIUS_DEG * PIXELS_PER_DEG )
RADIUS_BOUND = FRAME_SIZE // 3 - 3*RADIUS  # leave room for dots

def direction_to_velocity(degrees: float) -> NDArray:
    # pixels per second based on degrees/sec
    pixels_per_sec = DEFAULT_VELOCITY * PIXELS_PER_DEG

    # pixels per frame = pixels/sec / frames/sec
    pixels_per_frame = pixels_per_sec / FRAME_RATE

    # direction vector (unit)
    radians = np.deg2rad((degrees + 90) % 360)
    return np.array([np.cos(radians), np.sin(radians)]) * pixels_per_frame

def update_positions_with_circle(dots, velocities, center, radius, is_coherent, coherent_dir, directions):
    for i, dot in enumerate(dots):
        new_pos = dot.position + velocities[i]
        if np.linalg.norm(new_pos - center) > radius:
            new_pos = random_non_overlapping_position(velocities[i], dot.r, FRAME_SIZE, [d.position for d in dots],
                                                      center=center, radius_bound=radius)
            if is_coherent[i]:
                d = coherent_dir[i] if coherent_dir[i] is not None else choose_coherent_direction(directions)
                velocities[i] = direction_to_velocity(d)
            else:
                velocities[i] = velocity_for_random_direction()
        dot.position = new_pos


def create_axis_markers(frame_size: int, r: int, angle_deg: float | None = None, margin: int = 2) -> list[Dot]:
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    c = np.array([frame_size/2.0, frame_size/2.0], dtype=float)

    # center ON the circle (change to RADIUS_BOUND + r for tangent-outside)
    rho = RADIUS_BOUND + r + RADIUS

    positions = [c + rho*u, c - rho*u, c + rho*v, c - rho*v]

    patch = np.full((2*r, 2*r), 1, int)
    return [Dot(r=r, position=pos, fill=patch) for pos in positions]


def sample_life(lifetime_frames=LIFETIME_FRAMES) -> int:
    if lifetime_frames is None:
        return 0  # unlimited
    if isinstance(lifetime_frames, int):
        return max(1, lifetime_frames)
    lo, hi = lifetime_frames
    return int(np.random.randint(max(1, lo), max(lo + 1, hi + 1)))

def get_patches_for_radius(r: int): # Calculate matrices for visible and hidden per radius
    if r not in PATCH_VISIBLE:
        PATCH_VISIBLE[r] = np.full((2*r, 2*r), float(-1), dtype=float)
        PATCH_HIDDEN[r]  = np.ones((2*r, 2*r))
    return PATCH_VISIBLE[r], PATCH_HIDDEN[r]


def random_non_overlapping_position(
    v: NDArray,
    r: int,
    frame_size: int,
    existing_positions: list[np.ndarray],
    min_dist: float = None,
    max_attempts: int = 1000,
    center: np.ndarray = CENTER,
    radius_bound: int = RADIUS_BOUND
) -> np.ndarray:
    if min_dist is None:
        min_dist = 2*r
    for _ in range(max_attempts):
        pos = np.random.randint(0, frame_size, size=2)
        # ensure inside circle AND not too close to other dots
        if (np.linalg.norm(pos - center) <= radius_bound and
            np.linalg.norm(pos - center + v*LIFE_TIME_AVERAGE ) <= radius_bound 
            and all(np.linalg.norm(pos - p) >= min_dist for p in existing_positions)
        ):
            return pos
    # fallback: just put it somewhere valid inside circle
    while True:
        pos = np.random.randint(0, frame_size, size=2)
        if (np.linalg.norm(pos - center) <= radius_bound and
            np.linalg.norm(pos - center + v*LIFE_TIME_AVERAGE ) <= radius_bound ):
            return pos
    #return np.random.randint(0, frame_size, size=2)

def velocity_for_random_direction() -> NDArray:
    angle_deg = float(np.random.uniform(0.0, 360.0))
    return direction_to_velocity(angle_deg)

def choose_coherent_direction(directions=DIRECTIONS) -> int:
    return int(np.random.choice(directions))

def build_coherence_mask(num_dots: int, coherence: float) -> list[bool]:
    c = float(np.clip(coherence, 0.0, 1.0))
    k = int(np.round(c * num_dots))
    k = max(0, min(num_dots, k))
    mask = np.array([True]*k + [False]*(num_dots - k))
    np.random.shuffle(mask)
    return mask.tolist()


def generate_dots_with_properties(
    r: int,
    directions,
    direction_to_frequency: dict = None,
    visible=True,
    coherence: float = COHERENCE,
) -> tuple[list[Dot], list[NDArray], list[int], list[int], list[int], list[bool], list[int | None], list[int]]:
    if direction_to_frequency is None:
        direction_to_frequency = {
            d: FREQUENCIES[i % len(FREQUENCIES)]
            for i, d in enumerate(directions)
        }

    result_dots: list[Dot] = []
    velocities: list[NDArray] = []
    frequencies: list[int] = []
    init_lifetimes: list[int] = []
    life_left: list[int] = []
    is_coherent: list[bool] = []
    coherent_dir: list[int | None] = []
    phase_offsets: list[int] = []

    positions: list[np.ndarray] = []
    max_attempts = 5000

    # build coherence mask
    coh_mask = build_coherence_mask(NUMBER_OF_DOTS, coherence)

    # distribute coherent dots across directions (round-robin)
    coherent_dirs_cycle = cycle(directions)

    attempts = 0
    while len(result_dots) < NUMBER_OF_DOTS and attempts < max_attempts:
        
        coh = coh_mask[len(result_dots)]
        if coh:
            d = next(coherent_dirs_cycle)
            v = direction_to_velocity(d)
            f = int(direction_to_frequency[d])
            coh_d = d
        else:
            v = velocity_for_random_direction()
            f = int(np.random.choice(FREQUENCIES))
            coh_d = None
        
        while True:
            pos = np.random.randint(0, FRAME_SIZE, size=2)
            if (np.linalg.norm(pos - CENTER) <= RADIUS_BOUND and
                np.linalg.norm(pos - CENTER + v*LIFE_TIME_AVERAGE ) <= RADIUS_BOUND):
                break

        vis_patch, hid_patch = get_patches_for_radius(r)

        dir_index = directions.index(coh_d) if coh_d in directions else 0
        phase_offset = dir_index % 2

        dot = Dot(
            r=r,
            position=pos,
            fill=vis_patch if (phase_offset == 0) else hid_patch
        )

        result_dots.append(dot)
        velocities.append(v)
        frequencies.append(f)
        positions.append(pos)

        life = sample_life()
        init_lifetimes.append(life)
        life_left.append(life)

        is_coherent.append(coh)
        coherent_dir.append(coh_d)
        phase_offsets.append(phase_offset)

        attempts += 1
    return result_dots, velocities, frequencies, init_lifetimes, life_left, is_coherent, coherent_dir, phase_offsets

def is_visible(t: int, frequency: int, phase_offset: int = 0) -> bool:
    if frequency <= 0:
        return True

    cycle_len = FRAME_RATE // frequency
    if cycle_len <= 1:
        return True

    half_cycle = cycle_len // 2
    phase = (t + phase_offset * half_cycle) % cycle_len
    return phase < half_cycle

def generate_frames(
    dots: List[Dot],
    velocities: List[NDArray],
    frequencies: List[int],
    life_left: List[int],
    init_life: List[int],
    is_coherent: List[bool],
    coherent_dir: List[int | None],
    directions: List[float],
    phase_offsets: List[int],
    total_frames: int = EXPERIMENT_TIME,
    direction_to_frequency: dict | None = None,
) -> list[list[Dot]]:
    if direction_to_frequency is None:
        direction_to_frequency = {}

    for i in range(len(directions)):
        direction_to_frequency[directions[i]] = frequencies[i]

    frames: list[list[Dot]] = []

    for t in range(total_frames):
        # flicker dots based on their frequencies
        for i, dot in enumerate(dots):
            vis_patch, hid_patch = get_patches_for_radius(dot.r)
            visible = is_visible(t, frequencies[i], phase_offsets[i])
            dot.fill = vis_patch if visible else hid_patch

        # snapshot (deep copy of current dots state)
        frames.append([Dot(d.r, d.position.copy(), d.fill) for d in dots])

        # update next frame positions
        update_positions_with_circle(
            dots,
            velocities,
            CENTER,
            RADIUS_BOUND,
            is_coherent,
            coherent_dir,
            directions
        )

        # update life left and respawn dots if needed
        for i, dot in enumerate(dots):
            if life_left[i] > 0:
                life_left[i] -= 1
                if life_left[i] == 0:
                    
                    if is_coherent[i]:
                        d = coherent_dir[i] if coherent_dir[i] is not None else choose_coherent_direction(directions)
                        v = direction_to_velocity(d)
                        f = frequencies[i]
                        coherent_dir[i] = d
                    else:
                        v = velocity_for_random_direction()
                        f = frequencies[i]
                        coherent_dir[i] = None

                    new_pos = random_non_overlapping_position(
                        v, dot.r, FRAME_SIZE, [d.position for d in dots]
                    )

                    vis_patch, _ = get_patches_for_radius(dot.r)
                    dots[i] = Dot(r=dot.r, position=new_pos, fill=vis_patch)
                    velocities[i] = v
                    frequencies[i] = f
                    life_left[i] = init_life[i] = sample_life()

    result = [f + create_axis_markers(FRAME_SIZE, RADIUS, directions[0]) for f in frames]

    return result

def circular_mask(frame_size: int, radius: int, center: np.ndarray = CENTER) -> NDArray:
    Y, X = np.ogrid[:frame_size, :frame_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = np.zeros((frame_size, frame_size))
    mask[dist_from_center <= radius] = 1
    return mask

def generate_trials(
    direction_choice : str,
    num_trials: int = 105,
    dots_per_trial: int = NUMBER_OF_DOTS,
    coherence_levels: list[float] = None,
    r: int = RADIUS,
    total_frames: int = EXPERIMENT_TIME,
) -> list[tuple[list, float, float]]:
    """
    Returns a list of trials, each trial is a tuple: (frames, coherence, direction)
    """
    if coherence_levels is None:
        # Seven levels: 80, 80/sqrt(2), 80/sqrt(2)/sqrt(2), ... down to ~10
        base = 80
        coherence_levels = [base / (np.sqrt(2) ** i) for i in range(7)]
    
    # Repeat each level 15 times
    coherence_list = []
    for level in coherence_levels:
        coherence_list.extend([level / 100.0] * (num_trials // len(coherence_levels)))
    
    np.random.shuffle(coherence_list)  # randomize trial order

    all_trials = []

    for trial_idx in range(num_trials):
        coherence = coherence_list[trial_idx]
        if direction_choice == "random":
            direction = float(np.random.uniform(0.0, 360.0))
        elif direction_choice == "fixed":
            direction = random.choice([0,90,180,270])
        random_direction = [direction]

        # generate dots
        dots, velocities, freqs, init_life, life_left, is_coh, coh_dir, phase_offsets = generate_dots_with_properties(
            r, coherence=coherence, directions=random_direction
        )

        # generate frames
        trial_frames = generate_frames(
            dots, velocities, freqs, life_left, init_life,
            is_coh, coh_dir, random_direction, phase_offsets,
            total_frames=total_frames
        )

        all_trials.append((trial_frames, coherence, np.deg2rad(direction)))
        print(f"Trial {trial_idx+1}/{num_trials} done: coherence={coherence:.3f}, direction={direction:.1f}")

    return all_trials
