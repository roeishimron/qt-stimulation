import copy
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
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
    """
    Calculates a dot's lifetime in frames, ensuring it's a multiple of
    its cycle_time.
    """
    if cycle_time == 0:
        return max(1, int(np.random.exponential(scale=mean_lifetime)))
    mean_lifetime_in_cycles = mean_lifetime / cycle_time
    num_cycles = np.random.exponential(scale=mean_lifetime_in_cycles)
    num_cycles = max(1, int(round(num_cycles)))
    return num_cycles * cycle_time


def _create_coordinate_grid(
    stimulus_size_px: int,
    step: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a meshgrid of coordinates for the stimulus area."""
    x = np.arange(0, stimulus_size_px, step)
    y = np.arange(0, stimulus_size_px, step)
    # y is imaginary, x is real. Matches complex plane convention in this file.
    # Note: center is (size/2, size/2).
    # Meshgrid: X[i, j] is x-coord, Y[i, j] is y-coord.
    return np.meshgrid(x, y)


def _update_occupied_mask(
    occupied_mask: np.ndarray,
    position: complex,
    dot_radius: int,
    coordinate_grid: Tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Updates the occupied mask by blocking the area around the given position.
    """
    X, Y = coordinate_grid
    # Exclusion radius: prevent overlap. using 2*radius as safety.
    # User requirement: "remove its 2*radius around its center"
    exclusion_radius_sq = (2 * dot_radius) ** 2

    # Position might be float, grid is discrete.
    # Distance squared from position
    dist_sq = (X - position.real)**2 + (Y - position.imag)**2
    occupied_mask[dist_sq < exclusion_radius_sq] = True


def _find_valid_position_grid(
    occupied_mask: np.ndarray,
    velocity: complex,
    cycle_time: int,
    outer_boundary_radius: float,
    center: complex,
    coordinate_grid: Tuple[np.ndarray, np.ndarray]
) -> Tuple[complex, int]:
    """
    Finds a valid position using grid-based boolean masking.
    Returns (position, chosen_amount_of_cycles).
    """
    X, Y = coordinate_grid

    # 1. Base valid area: Inside the stimulus circle
    dist_from_center_sq = (
        (X - center.real)**2 + (Y - center.imag)**2
    )
    base_valid_mask = dist_from_center_sq <= outer_boundary_radius**2

    v_abs = abs(velocity)

    # Determine max_cycles
    if cycle_time > 0 and v_abs > 1e-9:
        # Distance to cross diameter = 2 * R
        # Max distance = 2 * outer_boundary_radius
        # Each cycle = v_abs * cycle_time
        max_cycles = int(2 * outer_boundary_radius / (v_abs * cycle_time))
        if max_cycles < 1:
            max_cycles = 1
    else:
        max_cycles = 1

    # Randomly select initial amount of cycles
    if cycle_time > 0 and v_abs > 1e-9:
        start_cycles = np.random.randint(1, max_cycles + 1)
        cycle_range = range(start_cycles, 0, -1)
    else:
        cycle_range = range(1, 0, -1)  # Just 1 iteration

    for amount_of_cycles in cycle_range:
        total_time = amount_of_cycles * cycle_time

        # Calculate valid intersection
        c2 = center - velocity * total_time
        dist_from_c2_sq = (X - c2.real)**2 + (Y - c2.imag)**2

        # Valid start region & remove occupied spots
        intersection_mask = base_valid_mask & (
            dist_from_c2_sq <= outer_boundary_radius**2
        )
        available_mask = intersection_mask & (~occupied_mask)

        # Get candidate indices
        candidate_indices = np.argwhere(available_mask)

        if len(candidate_indices) > 0:
            # Pick random one
            idx = np.random.randint(len(candidate_indices))
            y_idx, x_idx = candidate_indices[idx]

            # Convert to position (using grid coordinates)
            pos_x = X[y_idx, x_idx]
            pos_y = Y[y_idx, x_idx]
            return complex(pos_x, pos_y), amount_of_cycles

    raise RuntimeError(
        f"Could not find a valid position for dot with velocity {velocity} "
        "even after reducing cycles."
    )


def _place_dot_on_grid(
    velocity: complex,
    cycle_time: int,
    mean_lifetime: int,
    outer_boundary_radius: float,
    center: complex,
    occupied_mask: np.ndarray,
    coordinate_grid: Tuple[np.ndarray, np.ndarray],
    dot_radius: int,
    current_frame: int,
    dot: Dot | None = None,
    is_coherent: bool = False,
    visible_color: int = 0
) -> Dot:
    """Helper to place a dot on the grid and update the mask."""
    position, amount_of_cycles = _find_valid_position_grid(
        occupied_mask, velocity, cycle_time,
        outer_boundary_radius, center, coordinate_grid
    )

    _update_occupied_mask(
        occupied_mask, position, dot_radius,
        coordinate_grid
    )

    if cycle_time > 0:
        death_frame = current_frame + amount_of_cycles * cycle_time - 1
    else:
        frames_to_exit = _calculate_frames_to_exit(
            position - center, velocity, outer_boundary_radius
        )
        lifetime = _get_lifetime_duration(mean_lifetime, 0)
        death_frame = current_frame + min(lifetime, frames_to_exit) - 1

    if dot is None:
        return Dot(
            position=position,
            r=dot_radius,
            velocity=velocity,
            death_frame=death_frame,
            is_coherent=is_coherent,
            is_visible=True,
            visible_color=visible_color,
            cycle_time=cycle_time,
            placement_radius=outer_boundary_radius
        )
    else:
        dot.position = position
        dot.death_frame = death_frame
        dot.needs_replacement = False
        return dot


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
        markers.append(
            Dot(
                position=center + rel_pos,
                r=dot_radius,
                velocity=0j,
                death_frame=duration,
                is_coherent=False,
                visible_color=1,
                is_visible=True,
                cycle_time=0,
                placement_radius=placement_radius
            )
        )
    return markers


def _calculate_frames_to_exit(
    position: complex,
    velocity: complex,
    radius: float
) -> int:
    """Calculates number of frames until the dot exits the radius."""
    v_abs_sq = velocity.real**2 + velocity.imag**2
    if v_abs_sq < 1e-9:
        return 1_000_000  # Effectively infinite

    # Quadratic equation: |P + tV|^2 = R^2
    # |P|^2 + 2t(P.V) + t^2|V|^2 = R^2
    # a*t^2 + b*t + c = 0
    a = v_abs_sq
    # Dot product of position and velocity
    b = 2 * (position.real * velocity.real + position.imag * velocity.imag)
    c = abs(position)**2 - radius**2

    delta = b**2 - 4*a*c
    if delta < 0:
        return 1_000_000  # Should not happen if inside

    # We want the positive root for forward time
    t = (-b + np.sqrt(delta)) / (2 * a)
    return int(t)


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
    center: complex,
    stimulus_size_px: int,
    grid_step: float = 1.0
) -> List[List[Dot]]:
    """
    The core simulation engine for generating frames of moving dots.
    """
    outer_boundary_radius = stimulus_radius - dot_radius
    coordinate_grid = _create_coordinate_grid(stimulus_size_px, step=grid_step)

    # 1. Prepare Dot Properties
    dot_properties: List[Tuple[complex, bool, int, int]] = []
    zip_iter = zip(
        directions, direction_proportions, cycle_times, colors
    )
    for direction, proportion, cycle_time, color in zip_iter:
        count = int(round(proportion * num_dots))
        for _ in range(count):
            coherence = True if direction is not None else False
            chosen_direction = (
                direction if direction is not None
                else np.random.rand() * 2 * np.pi
            )
            velocity = motion_velocity * np.exp(
                1j * (chosen_direction + np.pi / 2)
            )
            dot_properties.extend(
                [(velocity, coherence, int(cycle_time), color)]
            )
    np.random.shuffle(dot_properties)

    # 2. Initial Placement
    dots: List[Dot] = []
    occupied_mask = np.zeros_like(coordinate_grid[0], dtype=bool)

    for velocity, is_coherent, cycle_time_int, color in dot_properties:
        dots.append(_place_dot_on_grid(
            velocity=velocity,
            cycle_time=cycle_time_int,
            mean_lifetime=mean_lifetime,
            outer_boundary_radius=outer_boundary_radius,
            center=center,
            occupied_mask=occupied_mask,
            coordinate_grid=coordinate_grid,
            dot_radius=dot_radius,
            current_frame=0,
            is_coherent=is_coherent,
            visible_color=color
        ))

    all_frames = []
    for frame_num in range(duration):
        for dot in dots:
            dot.position += dot.velocity
            if frame_num > dot.death_frame:
                dot.needs_replacement = True

        # Replacement Logic
        dots_to_replace = [d for d in dots if d.needs_replacement]

        if dots_to_replace:
            # Rebuild occupied mask from stable dots
            stable_dots = [d for d in dots if not d.needs_replacement]
            occupied_mask.fill(False)
            for d in stable_dots:
                _update_occupied_mask(
                    occupied_mask, d.position, dot_radius,
                    coordinate_grid
                )

            # Place new dots
            for dot in dots_to_replace:
                if not dot.is_coherent:
                    dot.velocity = motion_velocity * np.exp(
                        1j * (np.random.rand() * 2 * np.pi)
                    )

                _place_dot_on_grid(
                    velocity=dot.velocity,
                    cycle_time=dot.cycle_time,
                    mean_lifetime=mean_lifetime,
                    outer_boundary_radius=outer_boundary_radius,
                    center=center,
                    occupied_mask=occupied_mask,
                    coordinate_grid=coordinate_grid,
                    dot_radius=dot_radius,
                    current_frame=frame_num,
                    dot=dot
                )

        visible_dots_this_frame = [
            copy.deepcopy(d) for d in dots if (
                d.cycle_time == 0
            ) or ((frame_num % d.cycle_time) < (d.cycle_time / 2))
        ]
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
    display_markers: bool = False,
    grid_compression: float = 10.0
) -> Tuple[List[List[Dot]], List[Dot]]:
    """
    Generates frames of moving dots for a Random Dot Kinematogram (RDK).
    """
    total_proportion = sum((p.ratio for p in groups_properties))
    if not np.isclose(total_proportion, 1.0):
        raise ValueError(
            "The sum of proportions cannot be greater other than 1."
        )
    cycle_times = [p.cycle_time for p in groups_properties]
    if any((c % 2 != 0 for c in cycle_times)):
        raise ValueError(
            "cycle_time for coherent dots must be an even number."
        )

    stimulus_radius = stimulus_size_px / 2.0
    center = (stimulus_size_px / 2.0) + 1j * (stimulus_size_px / 2.0)

    if dot_radius % grid_compression != 0:
        raise ValueError(
            f"grid_compression ({grid_compression}) must divide "
            f"dot_radius ({dot_radius}) without remainder."
        )

    grid_step = dot_radius / grid_compression

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
        center=center,
        stimulus_size_px=stimulus_size_px,
        grid_step=grid_step
    )

    if not display_markers:
        return moving_dot_frames, []

    marker_direction = (
        groups_properties[0].direction
        if len(groups_properties) > 0 and groups_properties[0].direction
        else 0
    )
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
}


def test_dots_stay_in_bounds():
    """1. For every frame, all dots are inside the circle."""
    direction_props = [
        GroupProperties(
            ratio=1.0, direction=0, cycle_time=np.uint(0), color=-1
        )
    ]
    frames = generate_moving_dots(
        groups_properties=direction_props, **PARAMS
    )[0]
    stimulus_radius = PARAMS["stimulus_size_px"] / 2.0
    center = (PARAMS["stimulus_size_px"] / 2.0) * (1 + 1j)
    for i, frame in enumerate(frames):
        assert len(frame) > 0, f"Frame {i} should not be empty"
        for dot in frame:
            distance_from_center = abs(dot.position - center)
            assert distance_from_center <= stimulus_radius + 1e-9, (
                f"Dot is out of bounds in frame {i}"
            )


def test_dots_are_always_moving():
    """2. All dots move."""
    params = PARAMS.copy()

    # Create a "sandbox" with no boundaries and immortal dots to test motion
    # in isolation
    params["stimulus_size_px"] = 2000
    params["mean_lifetime"] = 1_000_000

    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=np.pi / 4,
            cycle_time=np.uint(0),
            color=-1
        )
    ]
    frames = generate_moving_dots(
        groups_properties=direction_props, **params
    )[0]
    for i in range(params["duration"] - 1):
        frame_t0 = [d for d in frames[i] if d.color == -1]
        frame_t1 = [d for d in frames[i+1] if d.color == -1]
        expected_positions_t1 = {
            dot.position + dot.velocity for dot in frame_t0
        }
        actual_positions_t1 = {dot.position for dot in frame_t1}
        num_moved_as_expected = len(
            expected_positions_t1.intersection(actual_positions_t1)
        )
        num_replaced = params["num_dots"] - num_moved_as_expected
        max_allowed_replacements = params["num_dots"] * 0.2
        assert num_replaced <= max_allowed_replacements, (
            f"Too many dots were replaced or didn't move correctly in "
            f"frame {i+1}"
        )


def test_jumps_only_on_cycle_end():
    """3. If there's no noise, jumps only happen on end-of-cycle."""
    params = PARAMS.copy()
    params["mean_lifetime"] = 30
    cycle_time = 12
    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(cycle_time),
            color=-1
        )
    ]
    frames = generate_moving_dots(
        groups_properties=direction_props, **params
    )[0]
    for i in range(params["duration"] - 1):
        frame_t0 = [d for d in frames[i] if d.color == -1]
        positions_t1 = {d.position for d in frames[i+1] if d.color == -1}
        expected_positions = {dot.position + dot.velocity for dot in frame_t0}
        jump_positions = positions_t1 - expected_positions
        if len(jump_positions) > 0:
            is_end_of_cycle = (i % cycle_time == cycle_time - 1)
            assert is_end_of_cycle, (
                f"A dot was replaced on frame {i}, which is not the end of "
                "a cycle."
            )


def test_visibility_with_zero_cycle_time():
    """4. If the cycle_time is 0, all dots are always visible."""
    num_dots = PARAMS["num_dots"]
    direction_props = [
        GroupProperties(
            ratio=0.5, direction=0, cycle_time=np.uint(0), color=-1
        ),
        GroupProperties(
            ratio=0.5, direction=np.pi, cycle_time=np.uint(0), color=-1
        )
    ]
    # This test implicitly uses noise_cycle_time=0 as well
    frames = generate_moving_dots(
        groups_properties=direction_props, **PARAMS
    )[0]
    for i, frame in enumerate(frames):
        moving_dots_in_frame = [dot for dot in frame if dot.color == -1]
        assert len(moving_dots_in_frame) == num_dots, (
            f"Expected {num_dots} visible dots in frame {i}, but found "
            f"{len(moving_dots_in_frame)}"
        )


@pytest.mark.parametrize(
    "test_id, direction_props",
    [
        ("100% Noise", [
            GroupProperties(
                ratio=1.0,
                direction=None,
                cycle_time=np.uint(0),
                color=-1
            )
        ]),
        ("Signal + Noise", [
            GroupProperties(
                ratio=0.5, direction=0, cycle_time=np.uint(10), color=-1
            ),
            GroupProperties(
                ratio=0.5,
                direction=None,
                cycle_time=np.uint(0),
                color=-1
            )
        ]),
        ("Two Opposing Signals", [
            GroupProperties(
                ratio=0.5, direction=0, cycle_time=np.uint(10), color=-1
            ),
            GroupProperties(
                ratio=0.5, direction=np.pi, cycle_time=np.uint(12), color=-1
            )
        ]),
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

    assert len(all_positions) > 0, (
        "Simulation produced no moving dots to analyze."
    )

    center_of_mass = np.mean(all_positions)

    tolerance = params["dot_radius"]
    assert np.isclose(center_of_mass, true_center, atol=tolerance), \
        f"Center of mass {center_of_mass} is biased for case '{test_id}'"


def test_ssvep_flicker():
    """
    6. SSVEP Flicker: Verify 10-frame cycle produces 5 visible /
    5 invisible frames.
    """
    cycle_time = 10
    duration = 30
    params = PARAMS.copy()
    params["duration"] = duration
    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(cycle_time),
            color=-1
        )
    ]

    frames, _ = generate_moving_dots(
        groups_properties=direction_props, **params
    )

    # Check visibility per frame
    visibility = []
    for frame in frames:
        # If the frame has dots with color != 0 (assuming invisible dots
        # might be filtered or have color 0)
        # Based on implementation, 'visible_dots_this_frame' only contains
        # dots if they are in the visible phase.
        # So checking if list is empty or not is sufficient if we assume
        # 100% coherence.
        # But let's check if any dot is visible.
        is_visible = len(frame) > 0
        visibility.append(is_visible)

    expected_visibility = []
    for i in range(duration):
        # Frame 0: 0 % 10 = 0 < 5 -> True
        # Frame 4: 4 % 10 = 4 < 5 -> True
        # Frame 5: 5 % 10 = 5 !< 5 -> False
        expected_visibility.append((i % cycle_time) < (cycle_time / 2))

    assert visibility == expected_visibility, (
        f"Visibility pattern mismatch.\n"
        f"Expected: {expected_visibility}\n"
        f"Actual:   {visibility}"
    )


def test_dots_stay_in_bounds_with_cycles():
    """7. Checks if dots stay in bounds when cycle_time > 0."""
    params = {
        "num_dots": 50,
        "dot_radius": 20,
        "stimulus_size_px": 500,
        "duration": 200,
        "motion_velocity": 5,  # Fast enough to exit
        "mean_lifetime": 200,  # Long life
    }

    cycle_time = 20
    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(cycle_time),
            color=-1
        )
    ]

    frames, _ = generate_moving_dots(
        groups_properties=direction_props, **params
    )

    stimulus_radius = params["stimulus_size_px"] / 2.0
    center = (params["stimulus_size_px"] / 2.0) * (1 + 1j)

    for i, frame in enumerate(frames):
        for dot in frame:
            # Check visible dots
            distance_from_center = abs(dot.position - center)
            assert distance_from_center <= stimulus_radius + 1e-9, (
                f"Frame {i}: Dot at {dot.position} is out of bounds "
                f"(dist={distance_from_center}, R={stimulus_radius})"
            )


@pytest.mark.parametrize("grid_compression", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_no_dot_overlaps(grid_compression):
    """
    8. Ensure no dots overlap (distance < 2r) in any frame.
    Uses two groups in the same direction with different cycle times.
    """
    dot_radius = 20
    params = {
        "num_dots": 30,
        "dot_radius": dot_radius,
        "stimulus_size_px": 500,
        "duration": 50,
        "motion_velocity": 5,
        "mean_lifetime": 100,
        "grid_compression": grid_compression
    }

    direction_props = [
        GroupProperties(
            ratio=0.5,
            direction=0,
            cycle_time=np.uint(2),
            color=-1
        ),
        GroupProperties(
            ratio=0.5,
            direction=0,
            cycle_time=np.uint(4),
            color=1
        )
    ]

    frames, _ = generate_moving_dots(
        groups_properties=direction_props, **params
    )

    min_dist = 2 * dot_radius

    for i, frame in enumerate(frames):
        # We only care about dots in the same frame
        positions = np.array([dot.position for dot in frame])
        num_dots_in_frame = len(positions)
        if num_dots_in_frame < 2:
            continue

        # Compute all-to-all distances
        # using complex numbers: |p1 - p2|
        for j in range(num_dots_in_frame):
            for k in range(j + 1, num_dots_in_frame):
                dist = abs(positions[j] - positions[k])
                assert dist >= min_dist - 1e-9, (
                    f"Frame {i}: Overlap detected between dot {j} and {k}. "
                    f"Distance: {dist}, Minimum allowed: {min_dist}"
                )


@pytest.mark.parametrize("grid_compression", [1.0, 2.0, 5.0, 10.0, 20.0])
def test_high_density_packing(grid_compression):
    """
    9. Verify that we can still fit a high number of dots even with
    coarse grids.
    For R=200, r=20, the theoretical limit is around 25 dots.
    We test if we can fit 20 dots.
    """
    dot_radius = 20
    params = {
        "num_dots": 20,
        "dot_radius": dot_radius,
        "stimulus_size_px": 400,
        "duration": 1,
        "motion_velocity": 0,
        "mean_lifetime": 100,
        "grid_compression": grid_compression
    }

    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(0),
            color=-1
        )
    ]

    # This should not raise RuntimeError if packing is efficient enough
    try:
        generate_moving_dots(
            groups_properties=direction_props, **params
        )
    except RuntimeError as e:
        pytest.fail(
            f"Failed to pack {params['num_dots']} dots with "
            f"grid_compression={grid_compression}. Error: {e}"
        )
