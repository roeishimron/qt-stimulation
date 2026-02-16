import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Tuple
import pytest

# ==============================================================================
# ## Simulation Code
# ==============================================================================

# Define the Dot data structure


@dataclass
class Dot:
    """Represents a single dot in the stimulus."""
    position: complex
    r: int
    velocity: complex
    death_frame: int
    is_coherent: bool
    is_visible: bool
    visible_color: int
    cycle_time: int
    placement_radius: float
    needs_replacement: bool = False

    @property
    def color(self) -> int:
        return self.is_visible * self.visible_color

    @property
    def x(self) -> float:
        """The x-coordinate of the dot."""
        return self.position.real

    @property
    def y(self) -> float:
        """The y-coordinate of the dot."""
        return self.position.imag

# Helper Functions


def _get_lifetime_duration(mean_lifetime: int, cycle_time: int) -> int:
    """Calculates a dot's lifetime in frames, ensuring it's a multiple of its cycle_time."""
    if cycle_time == 0:
        return max(1, int(np.random.exponential(scale=mean_lifetime)))
    mean_lifetime_in_cycles = mean_lifetime / cycle_time
    num_cycles = np.random.exponential(scale=mean_lifetime_in_cycles)
    num_cycles = max(1, int(round(num_cycles)))
    return num_cycles * cycle_time


def _find_valid_position(
    placement_radius: float,
    velocity: complex,
    dot_radius: int,
    cycle_time: int,
    existing_dots: List[Dot],
    center: complex,
    max_tries: int = 1000
) -> complex:
    """
    Finds a valid, non-overlapping, and spatially unbiased position by
    directly sampling angles centered around the upstream direction.
    """
    if placement_radius <= 0:
        return center

    placement_radius_sq = placement_radius**2
    c2 = -velocity * cycle_time

    # Angle pointing opposite to the velocity
    upstream_angle = np.angle(-velocity)

    for _ in range(max_tries):
        # 1. Generate angle centered around upstream direction with range pi
        angle_offset = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta = upstream_angle + angle_offset
        # Generate radius ensuring uniform area distribution
        r = placement_radius * np.sqrt(np.random.rand())
        rel_pos = r * np.exp(1j * theta)

        # 2. Check if it's in the second, shifted circle.
        if abs(rel_pos - c2)**2 >= placement_radius_sq:
            continue

        # 3. Check for overlaps
        candidate_pos = center + rel_pos
        is_overlapping = False
        for d in existing_dots:
            if not np.isclose(d.velocity, velocity):
                continue
            if abs(candidate_pos - d.position)**2 < (2 * dot_radius)**2:
                is_overlapping = True
                break
        if not is_overlapping:
            return candidate_pos

    # Fallback if a position isn't found
    return center


def _create_direction_markers(
    direction: float,
    dot_radius: int,
    center: complex,
    stimulus_radius: float,
    duration: int
) -> List[Dot]:
    """Creates four static dots as markers."""
    placement_radius = stimulus_radius - dot_radius
    base_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    relative_positions = placement_radius * np.exp(1j * base_angles)
    rotation_angle = direction + np.pi / 2
    rotation_vector = np.exp(1j * rotation_angle)
    rotated_relative_positions = relative_positions * rotation_vector
    markers = []
    for rel_pos in rotated_relative_positions:
        markers.append(Dot(position=center + rel_pos, r=dot_radius, velocity=0j, death_frame=duration,
                       is_coherent=False, visible_color=1, is_visible=True, cycle_time=0, placement_radius=placement_radius))
    return markers


def _simulate_moving_dots(
    num_dots: int,
    dot_radius: int,
    duration: int,
    direction_proportions: List[float],
    directions: List[float | None],
    cycle_times: List[np.uint],
    colors: List[int],
    motion_velocity: int,
    mean_lifetime: int,
    stimulus_radius: float,
    center: complex
) -> List[List[Dot]]:
    """
    The core simulation engine for generating frames of moving dots.
    """
    outer_boundary_radius = stimulus_radius - dot_radius

    dot_properties = []
    for direction, proportion, cycle_time, color in zip(directions, direction_proportions, cycle_times, colors):
        count = int(round(proportion * num_dots))
        for _ in range(count):
            coherence = True if direction is not None else False
            # should have been "direction if coherence" but pylint won't understand
            chosen_direction = direction if direction is not None else np.random.rand() * 2 * np.pi

            velocity = motion_velocity * np.exp(1j * (chosen_direction + np.pi / 2))
            dot_properties.extend(
                [(velocity, coherence, int(cycle_time), color)])
    np.random.shuffle(dot_properties)

    dots: List[Dot] = []
    for velocity, is_coherent, cycle_time, color in dot_properties:
        lifetime_duration = _get_lifetime_duration(mean_lifetime, cycle_time)
        death_frame = lifetime_duration
        position = _find_valid_position(
            outer_boundary_radius, velocity, dot_radius, cycle_time, dots, center)
        dots.append(Dot(position=position, r=dot_radius, velocity=velocity, death_frame=death_frame,
                    is_coherent=is_coherent, is_visible=True, visible_color=color, cycle_time=cycle_time,
                        placement_radius=outer_boundary_radius))

    all_frames = []
    for frame_num in range(duration):
        for dot in dots:
            dot.position += dot.velocity
            if frame_num > dot.death_frame or abs(dot.position - center) > dot.placement_radius:
                dot.needs_replacement = True

        dots_to_replace = [d for d in dots if d.needs_replacement and (
            (d.cycle_time == 0) or (frame_num % d.cycle_time == d.cycle_time - 1))]
        stable_dots = [d for d in dots if d not in dots_to_replace]

        current_obstacles = stable_dots[:]
        for dot in dots_to_replace:
            lifetime_duration = _get_lifetime_duration(
                mean_lifetime, dot.cycle_time)
            dot.death_frame = frame_num + lifetime_duration
            if not dot.is_coherent:
                dot.velocity = motion_velocity * \
                    np.exp(1j * (np.random.rand() * 2 * np.pi))

            dot.placement_radius = outer_boundary_radius - \
                (abs(dot.velocity) * dot.cycle_time)
            dot.position = _find_valid_position(
                dot.placement_radius, dot.velocity, dot.r, dot.cycle_time, current_obstacles, center)
            current_obstacles.append(dot)
            dot.needs_replacement = False

        visible_dots_this_frame = [copy.deepcopy(d) for d in dots if (
            d.cycle_time == 0) or ((frame_num % d.cycle_time) < (d.cycle_time / 2))]
        all_frames.append(visible_dots_this_frame)

    return all_frames


@dataclass
class GroupProperties:
    ratio: float
    direction: float | None
    cycle_time: np.uint
    color: int

# Main Public Function


def generate_moving_dots(
    num_dots: int,
    dot_radius: int,
    stimulus_size_px: int,
    duration: int,
    motion_velocity: int,
    mean_lifetime: int,
    groups_properties: List[GroupProperties] = [],
    display_markers: bool = False
) -> Tuple[List[List[Dot]], List[Dot]]:
    """
    Generates frames of moving dots for a Random Dot Kinematogram (RDK).
    """
    total_proportion = sum((p.ratio for p in groups_properties))
    if not np.isclose(total_proportion, 1.0):
        raise ValueError(
            "The sum of proportions cannot be greater other than 1.")
    cycle_times = [p.cycle_time for p in groups_properties]
    if any((c % 2 != 0 for c in cycle_times)):
        raise ValueError(
            "cycle_time for coherent dots must be an even number.")

    stimulus_radius = stimulus_size_px / 2.0
    center = (stimulus_size_px / 2.0) + 1j * (stimulus_size_px / 2.0)

    moving_dot_frames = _simulate_moving_dots(
        num_dots=num_dots,
        dot_radius=dot_radius,
        duration=duration,
        direction_proportions=[p.ratio for p in groups_properties],
        directions=[p.direction for p in groups_properties],
        cycle_times=[p.cycle_time for p in groups_properties],
        colors=[p.color for p in groups_properties],
        motion_velocity=motion_velocity,
        mean_lifetime=mean_lifetime,
        stimulus_radius=stimulus_radius,
        center=center
    )

    if not display_markers:
        return moving_dot_frames, []

    marker_direction = groups_properties[0].direction if len(
        groups_properties) > 0 and groups_properties[0].direction else 0
    markers = _create_direction_markers(
        marker_direction, dot_radius, center, stimulus_radius, duration
    )

    final_frames = [frame + markers for frame in moving_dot_frames]
    return final_frames, markers


# ==============================================================================
# ## Test Suite
# ==============================================================================


# Default parameters for tests
PARAMS = {
    "num_dots": 50,
    "dot_radius": 20,
    "stimulus_size_px": 500,
    "duration": 50,
    "motion_velocity": 2,
    "mean_lifetime": 60,
    "noise_cycle_time": 0,
}


def test_dots_stay_in_bounds():
    """1. For every frame, all dots are inside the circle."""
    direction_props = np.array([[0, 1.0, 10]])
    frames = generate_moving_dots(
        groups_properties=direction_props, **PARAMS)[0]
    stimulus_radius = PARAMS["stimulus_size_px"] / 2.0
    center = (PARAMS["stimulus_size_px"] / 2.0) * (1 + 1j)
    for i, frame in enumerate(frames):
        assert len(frame) > 0, f"Frame {i} should not be empty"
        for dot in frame:
            distance_from_center = abs(dot.position - center)
            assert distance_from_center <= stimulus_radius + \
                1e-9, f"Dot is out of bounds in frame {i}"


def test_dots_are_always_moving():
    """2. All dots move."""
    params = PARAMS.copy()

    # Create a "sandbox" with no boundaries and immortal dots to test motion in isolation
    params["stimulus_size_px"] = 1_000_000
    params["mean_lifetime"] = 1_000_000

    direction_props = np.array([[np.pi / 4, 1.0, 0]])
    frames = generate_moving_dots(
        groups_properties=direction_props, **params)[0]
    for i in range(params["duration"] - 1):
        frame_t0 = [d for d in frames[i] if d.color == -1]
        frame_t1 = [d for d in frames[i+1] if d.color == -1]
        expected_positions_t1 = {dot.position +
                                 dot.velocity for dot in frame_t0}
        actual_positions_t1 = {dot.position for dot in frame_t1}
        num_moved_as_expected = len(
            expected_positions_t1.intersection(actual_positions_t1))
        num_replaced = params["num_dots"] - num_moved_as_expected
        max_allowed_replacements = params["num_dots"] * 0.2
        assert num_replaced <= max_allowed_replacements, f"Too many dots were replaced or didn't move correctly in frame {i+1}"


def test_jumps_only_on_cycle_end():
    """3. If there's no noise, jumps only happen on end-of-cycle."""
    params = PARAMS.copy()
    params["mean_lifetime"] = 30
    cycle_time = 12
    direction_props = np.array([[0, 1.0, cycle_time]])
    frames = generate_moving_dots(
        groups_properties=direction_props, **params)[0]
    for i in range(params["duration"] - 1):
        frame_t0 = [d for d in frames[i] if d.color == -1]
        positions_t1 = {d.position for d in frames[i+1] if d.color == -1}
        expected_positions = {dot.position + dot.velocity for dot in frame_t0}
        jump_positions = positions_t1 - expected_positions
        if len(jump_positions) > 0:
            is_end_of_cycle = (i % cycle_time == cycle_time - 1)
            assert is_end_of_cycle, f"A dot was replaced on frame {i}, which is not the end of a cycle."


def test_visibility_with_zero_cycle_time():
    """4. If the cycle_time is 0, all dots are always visible."""
    num_dots = PARAMS["num_dots"]
    direction_props = np.array([[0, 0.5, 0], [np.pi, 0.5, 0]])
    # This test implicitly uses noise_cycle_time=0 as well
    frames = generate_moving_dots(
        groups_properties=direction_props, **PARAMS)[0]
    for i, frame in enumerate(frames):
        moving_dots_in_frame = [dot for dot in frame if dot.color == -1]
        assert len(
            moving_dots_in_frame) == num_dots, f"Expected {num_dots} visible dots in frame {i}, but found {len(moving_dots_in_frame)}"


@pytest.mark.parametrize(
    "test_id, direction_props",
    [
        ("100% Noise", np.zeros((0, 3))),
        ("Signal + Noise", np.array([[0, 0.5, 10]])),
        ("Two Opposing Signals", np.array([[0, 0.5, 10], [np.pi, 0.5, 12]])),
    ]
)
def test_spatial_distribution_is_unbiased(test_id, direction_props):
    """5. The average mass center is the geometric center (no spatial bias)."""
    params = PARAMS.copy()
    params["duration"] = 5000

    frames = generate_moving_dots(
        groups_properties=direction_props,
        **params
    )[0]

    true_center = (params["stimulus_size_px"] / 2.0) * (1 + 1j)

    all_positions = []
    for frame in frames:
        moving_dots = [d for d in frame if d.color == -1]
        all_positions.extend([d.position for d in moving_dots])

    assert len(all_positions) > 0, "Simulation produced no moving dots to analyze."

    center_of_mass = np.mean(all_positions)

    tolerance = params["dot_radius"]
    assert np.isclose(center_of_mass, true_center, atol=tolerance), \
        f"Center of mass {center_of_mass} is biased for case '{test_id}'"
