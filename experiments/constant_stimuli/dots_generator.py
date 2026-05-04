import copy
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple, Union
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
    max_cycles: int
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


def _get_lifetime_duration(max_cycles: int, cycle_time: int) -> int:
    """
    Calculates a dot's lifetime in frames, ensuring it's a multiple of
    its cycle_time.
    """
    if cycle_time == 0:
        return max(1, np.random.randint(0, max_cycles + 1))
    num_cycles = np.random.randint(0, max_cycles + 1)
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


def _get_cycle_range(
    max_cycles: int,
    cycle_time: int,
    velocity: complex,
    outer_boundary_radius: float
) -> range | List[int]:
    """Calculates the range of possible cycles for a dot's lifetime."""
    v_abs = abs(velocity)
    if cycle_time > 0 and v_abs > 1e-9:
        # Distance to cross diameter = 2 * R
        # Max distance = 2 * outer_boundary_radius
        # Each cycle = v_abs * cycle_time
        diameter_max_cycles = int(
            2 * outer_boundary_radius / (v_abs * cycle_time)
        )
        diameter_max_cycles = max(1, diameter_max_cycles)

        effective_max_cycles = min(max_cycles, diameter_max_cycles)
        if effective_max_cycles >= 1:
            start_cycles = np.random.randint(1, effective_max_cycles + 1)
        else:
            start_cycles = 0
        return range(start_cycles, -1, -1)
    return [0]


def _find_valid_position_grid(
    occupied_mask: np.ndarray,
    velocity: complex,
    cycle_time: int,
    outer_boundary_radius: float,
    center: complex,
    coordinate_grid: Tuple[np.ndarray, np.ndarray],
    max_cycles: int
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

    cycle_range = _get_cycle_range(
        max_cycles, cycle_time, velocity, outer_boundary_radius
    )

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


def _find_valid_position_random(
    velocity: complex,
    cycle_time: int,
    outer_boundary_radius: float,
    center: complex,
    group_dots: List[Dot],
    dot_radius: int,
    max_cycles: int,
    max_tries: int = 10
) -> Tuple[complex, int] | None:
    """Tries to find a valid position randomly within the intersection of circles."""
    cycle_range = _get_cycle_range(
        max_cycles, cycle_time, velocity, outer_boundary_radius
    )

    for amount_of_cycles in cycle_range:
        total_time = amount_of_cycles * cycle_time
        c2 = center - velocity * total_time
        # Intersection of two circles: Circle(center, R) and Circle(c2, R)

        for _ in range(max_tries):
            # Pick a random point in Circle(center, R)
            r = outer_boundary_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            pos = center + r * np.exp(1j * theta)

            # Check if it's also in Circle(c2, R)
            if abs(pos - c2) <= outer_boundary_radius:
                # Check overlap with existing dots in this group
                is_overlapping = False
                for other in group_dots:
                    if abs(pos - other.position) < 2 * dot_radius:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    return pos, amount_of_cycles
    return None


def _place_dot(
    velocity: complex,
    cycle_time: int,
    max_cycles: int,
    outer_boundary_radius: float,
    center: complex,
    coordinate_grid: Tuple[np.ndarray, np.ndarray],
    dot_radius: int,
    current_frame: int,
    group_dots: List[Dot],
    dot: Dot | None = None,
    is_coherent: bool = False,
    visible_color: int = 0
) -> Dot:
    """
    Places a dot using dual strategy: random-first then grid-based fallback.
    Adds to the group dot list if provided.
    """
    # 1. Try Random Placement
    res = _find_valid_position_random(
        velocity, cycle_time, outer_boundary_radius, center,
        group_dots, dot_radius, max_cycles
    )

    if res is not None:
        position, amount_of_cycles = res
    else:
        # 2. Fallback to Grid Placement
        # Create mask only when required
        occupied_mask = np.zeros_like(coordinate_grid[0], dtype=bool)
        for d in group_dots:
            _update_occupied_mask(
                occupied_mask, d.position, dot_radius,
                coordinate_grid
            )

        position, amount_of_cycles = _find_valid_position_grid(
            occupied_mask, velocity, cycle_time,
            outer_boundary_radius, center, coordinate_grid,
            max_cycles
        )

    if cycle_time > 0:
        death_frame = current_frame + amount_of_cycles * cycle_time - 1
    else:
        death_frame = _compute_death_frame_from_position(
            position, velocity, cycle_time, max_cycles,
            outer_boundary_radius, center, current_frame
        )

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
            max_cycles=max_cycles,
            placement_radius=outer_boundary_radius
        )
    else:
        dot.position = position
        dot.death_frame = death_frame
        dot.needs_replacement = False
        dot.velocity = velocity
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
                max_cycles=duration,
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


def _compute_death_frame_from_position(
    position: complex,
    velocity: complex,
    cycle_time: int,
    max_cycles: int,
    outer_boundary_radius: float,
    center: complex,
    current_frame: int,
) -> int:
    """
    Computes a fresh death_frame for a dot at a known (position, velocity).
    Returns current_frame - 1 when the dot can't survive even one step or one
    cycle, so the simulation's existing replacement path picks it up.
    """
    if abs(position - center) > outer_boundary_radius:
        return current_frame - 1

    frames_to_exit = _calculate_frames_to_exit(
        position - center, velocity, outer_boundary_radius
    )
    if cycle_time > 0:
        max_possible_cycles = max(0, frames_to_exit // cycle_time)
        effective_max = min(max_cycles, max_possible_cycles)
        amount_of_cycles = np.random.randint(0, effective_max + 1)
        return current_frame + amount_of_cycles * cycle_time - 1

    lifetime = _get_lifetime_duration(max_cycles, 0)
    return current_frame + min(lifetime, frames_to_exit) - 1


def _initialize_dots(
    num_dots: int,
    dot_radius: int,
    motion_velocity: int,
    direction_proportions: List[float],
    directions: List[float | None],
    cycle_times: List[np.uint],
    colors: List[int],
    max_cycles: List[int],
    outer_boundary_radius: float,
    center: complex,
) -> Tuple[List[Dot], dict[complex, List[Dot]]]:
    """
    Synthesize a starting list of "dead" dots (death_frame=-1, position=center).
    The main-loop replacement path places them on frame 0. Returns (dots, empty
    velocity_groups).
    """
    dot_properties: List[Tuple[complex, bool, int, int, int]] = []
    zip_iter = zip(
        directions, direction_proportions, cycle_times, colors, max_cycles
    )
    for direction, proportion, cycle_time, color, m_cycles in zip_iter:
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
            dot_properties.append(
                (velocity, coherence, int(cycle_time), color, m_cycles)
            )
    np.random.shuffle(dot_properties)

    dots: List[Dot] = []
    for velocity, is_coherent, cycle_time_int, color, m_cycles in dot_properties:
        dots.append(Dot(
            position=center,
            r=dot_radius,
            velocity=velocity,
            death_frame=-1,
            is_coherent=is_coherent,
            is_visible=True,
            visible_color=color,
            cycle_time=cycle_time_int,
            max_cycles=m_cycles,
            placement_radius=outer_boundary_radius,
        ))
    velocity_groups: dict[complex, List[Dot]] = {}
    return dots, velocity_groups


def _advance_frame(
    dots: List[Dot],
    frame_num: int,
    motion_velocity: int,
    dot_radius: int,
    outer_boundary_radius: float,
    center: complex,
    coordinate_grid: Tuple[np.ndarray, np.ndarray],
    velocity_groups: dict[complex, List[Dot]],
) -> List[Dot]:
    """One frame step: advance positions, run replacement for dead dots,
    return visible dots. Mutates `dots` and `velocity_groups`."""
    def get_group(v: complex) -> List[Dot]:
        if v not in velocity_groups:
            velocity_groups[v] = []
        return velocity_groups[v]

    for dot in dots:
        dot.position += dot.velocity
        if frame_num > dot.death_frame:
            dot.needs_replacement = True

    dots_to_replace = [d for d in dots if d.needs_replacement]
    if dots_to_replace:
        for g_dots in velocity_groups.values():
            g_dots.clear()
        for d in dots:
            if not d.needs_replacement:
                get_group(d.velocity).append(d)
        for dot in dots_to_replace:
            if not dot.is_coherent:
                dot.velocity = motion_velocity * np.exp(
                    1j * (np.random.rand() * 2 * np.pi)
                )
            group_dots = get_group(dot.velocity)
            _place_dot(
                velocity=dot.velocity,
                cycle_time=dot.cycle_time,
                max_cycles=dot.max_cycles,
                outer_boundary_radius=outer_boundary_radius,
                center=center,
                coordinate_grid=coordinate_grid,
                dot_radius=dot_radius,
                current_frame=frame_num,
                group_dots=group_dots,
                dot=dot,
            )
            group_dots.append(dot)

    return [
        copy.deepcopy(d) for d in dots
        if d.cycle_time == 0
        or (frame_num % d.cycle_time) < (d.cycle_time / 2)
    ]


def _simulate_moving_dots(
    num_dots: int,
    dot_radius: int,
    duration: int,
    direction_proportions: List[float],
    directions: List[float | None],
    cycle_times: List[np.uint],
    colors: List[int],
    motion_velocity: int,
    max_cycles: List[int],
    stimulus_radius: float,
    center: complex,
    stimulus_size_px: int,
    grid_step: float = 1.0,
) -> Tuple[List[List[Dot]], List[Dot]]:
    """The core simulation engine. Returns (visible_frames, final_dots).
    `final_dots` is the full last-frame dot state including dots that were
    invisible in the last cycle phase — required for chained simulations."""
    outer_boundary_radius = stimulus_radius - dot_radius
    coordinate_grid = _create_coordinate_grid(stimulus_size_px, step=grid_step)
    dots, velocity_groups = _initialize_dots(
        num_dots, dot_radius, motion_velocity,
        direction_proportions, directions, cycle_times, colors, max_cycles,
        outer_boundary_radius, center,
    )
    frames = [
        _advance_frame(
            dots, frame_num, motion_velocity, dot_radius,
            outer_boundary_radius, center, coordinate_grid, velocity_groups,
        )
        for frame_num in range(duration)
    ]
    return frames, [copy.deepcopy(d) for d in dots]


@dataclass
class GroupProperties:
    ratio: float
    direction: float | None
    cycle_time: np.uint
    color: int
    max_cycles: int


class Incoherent:
    """Sentinel: random per-dot direction in this group."""
    pass


INCOHERENT = Incoherent()


@dataclass
class GroupModification:
    """Specifies how a fraction of seed dots should be modified for a
    chained simulation. Only direction and color may change.

    direction:
      - float       → fixed angle (radians); dot becomes coherent
      - INCOHERENT  → randomize per-dot direction; dot becomes incoherent
      - None        → preserve existing dot.velocity and dot.is_coherent
    """
    ratio: float
    direction: Union[float, Incoherent, None]
    color: int


@dataclass(frozen=True)
class ContinuationContext:
    """Opaque continuation state. Pass it back into `continue_moving_dots`;
    callers don't need to read its fields directly.

    Construction enforces the chaining invariant: the simulation that
    produced this context must end at a clean cycle boundary, i.e.
    `duration` is a multiple of every non-zero `cycle_time`. This is the
    only condition under which a chained simulation preserves both the
    visibility flicker phase and masks any frame-0 spatial replacement
    behind the natural invisible→visible transition. Construction raises
    `ValueError` if the invariant is violated."""
    last_dot_frame: List[Dot]
    motion_velocity: int
    dot_radius: int
    stimulus_size_px: int
    grid_compression: float
    duration: int
    cycle_times: Tuple[int, ...]

    def __post_init__(self):
        for c in self.cycle_times:
            if c != 0 and self.duration % c != 0:
                raise ValueError(
                    f"duration ({self.duration}) must be a multiple of "
                    f"every non-zero cycle_time so the visibility wave "
                    f"aligns at the seam; cycle_time={c} violates this."
                )


class GenerationResult(NamedTuple):
    frames: List[List[Dot]]                       # display-ready (dots + markers)
    markers: List[Dot]
    continuation: Optional[ContinuationContext]   # None when chaining invariant fails


# Main Public Function


def generate_moving_dots(
    num_dots: int,
    dot_radius: int,
    stimulus_size_px: int,
    duration: int,
    motion_velocity: int,
    groups_properties: List[GroupProperties] = [],
    display_markers: bool = False,
    grid_compression: float = 10.0,
) -> GenerationResult:
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

    for p in groups_properties:
        if p.cycle_time > 0 and motion_velocity > 0:
            actual_max = stimulus_size_px // motion_velocity // p.cycle_time
            if p.max_cycles > actual_max:
                raise ValueError(
                    f"max_cycles {p.max_cycles} is larger than the actual "
                    f"maximum {actual_max} (stimulus_size_px // "
                    f"motion_velocity // cycle_time)"
                )

    stimulus_radius = stimulus_size_px / 2.0
    center = (stimulus_size_px / 2.0) + 1j * (stimulus_size_px / 2.0)

    if dot_radius % grid_compression != 0:
        raise ValueError(
            f"grid_compression ({grid_compression}) must divide "
            f"dot_radius ({dot_radius}) without remainder."
        )

    grid_step = dot_radius / grid_compression

    moving_dot_frames, final_dots = _simulate_moving_dots(
        num_dots=num_dots,
        dot_radius=dot_radius,
        duration=duration,
        direction_proportions=[p.ratio for p in groups_properties],
        directions=[p.direction for p in groups_properties],
        cycle_times=[p.cycle_time for p in groups_properties],
        colors=[p.color for p in groups_properties],
        motion_velocity=motion_velocity,
        max_cycles=[p.max_cycles for p in groups_properties],
        stimulus_radius=stimulus_radius,
        center=center,
        stimulus_size_px=stimulus_size_px,
        grid_step=grid_step,
    )

    markers: List[Dot] = []
    if display_markers:
        marker_direction = (
            groups_properties[0].direction
            if len(groups_properties) > 0 and groups_properties[0].direction
            else 0
        )
        markers = _create_direction_markers(
            marker_direction, dot_radius, center, stimulus_radius, duration
        )

    frames = [frame + markers for frame in moving_dot_frames]
    try:
        continuation: Optional[ContinuationContext] = ContinuationContext(
            last_dot_frame=final_dots,
            motion_velocity=motion_velocity,
            dot_radius=dot_radius,
            stimulus_size_px=stimulus_size_px,
            grid_compression=grid_compression,
            duration=duration,
            cycle_times=tuple(int(p.cycle_time) for p in groups_properties),
        )
    except ValueError:
        continuation = None
    return GenerationResult(
        frames=frames, markers=markers, continuation=continuation,
    )


def _apply_modification_to_dot(
    dot: Dot,
    mod: GroupModification,
    motion_velocity: int,
    outer_boundary_radius: float,
    center: complex,
) -> None:
    """Mutate a single dot in place per the modification spec."""
    if isinstance(mod.direction, Incoherent):
        dot.velocity = motion_velocity * np.exp(
            1j * (np.random.rand() * 2 * np.pi + np.pi / 2)
        )
        dot.is_coherent = False
    elif mod.direction is None:
        pass  # preserve velocity + coherence
    else:
        dot.velocity = motion_velocity * np.exp(
            1j * (float(mod.direction) + np.pi / 2)
        )
        dot.is_coherent = True
    dot.visible_color = mod.color
    dot.needs_replacement = False
    dot.death_frame = _compute_death_frame_from_position(
        dot.position, dot.velocity, dot.cycle_time, dot.max_cycles,
        outer_boundary_radius, center, current_frame=0,
    )


def continue_moving_dots(
    previous: GenerationResult,
    duration: int,
    modifications: List[GroupModification],
) -> GenerationResult:
    """Continue a previous simulation. Only direction and color may change
    via `modifications`; cycle_time, max_cycles, motion_velocity, and the
    stimulus geometry are inherited from `previous`."""
    if previous.continuation is None:
        raise ValueError(
            "previous has no continuation context — its duration was not a "
            "multiple of every non-zero cycle_time, so chaining would break "
            "the visibility flicker phase."
        )
    ctx = previous.continuation
    if not ctx.last_dot_frame:
        raise ValueError("previous has no dots to continue from")
    if not np.isclose(sum(m.ratio for m in modifications), 1.0):
        raise ValueError("modifications ratios must sum to 1.0")

    stimulus_radius = ctx.stimulus_size_px / 2.0
    center = stimulus_radius * (1 + 1j)
    outer_boundary_radius = stimulus_radius - ctx.dot_radius
    grid_step = ctx.dot_radius / ctx.grid_compression
    coordinate_grid = _create_coordinate_grid(
        ctx.stimulus_size_px, step=grid_step
    )

    seed_dots = [copy.deepcopy(d) for d in ctx.last_dot_frame]
    np.random.shuffle(seed_dots)

    velocity_groups: dict[complex, List[Dot]] = {}
    idx = 0
    for mod in modifications:
        count = int(round(mod.ratio * len(seed_dots)))
        for dot in seed_dots[idx:idx + count]:
            _apply_modification_to_dot(
                dot, mod, ctx.motion_velocity,
                outer_boundary_radius, center,
            )
            if dot.death_frame >= 0:
                velocity_groups.setdefault(dot.velocity, []).append(dot)
        idx += count

    moving_dot_frames = [
        _advance_frame(
            seed_dots, frame_num, ctx.motion_velocity, ctx.dot_radius,
            outer_boundary_radius, center, coordinate_grid, velocity_groups,
        )
        for frame_num in range(duration)
    ]

    markers = previous.markers
    try:
        new_ctx: Optional[ContinuationContext] = ContinuationContext(
            last_dot_frame=[copy.deepcopy(d) for d in seed_dots],
            motion_velocity=ctx.motion_velocity,
            dot_radius=ctx.dot_radius,
            stimulus_size_px=ctx.stimulus_size_px,
            grid_compression=ctx.grid_compression,
            duration=duration,
            cycle_times=ctx.cycle_times,
        )
    except ValueError:
        new_ctx = None
    return GenerationResult(
        frames=[f + markers for f in moving_dot_frames],
        markers=markers,
        continuation=new_ctx,
    )


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
}
DEFAULT_MAX_CYCLES = 20


def test_dots_stay_in_bounds():
    """1. For every frame, all dots are inside the circle."""
    direction_props = [
        GroupProperties(
            ratio=1.0, direction=0, cycle_time=np.uint(0), color=-1,
            max_cycles=DEFAULT_MAX_CYCLES
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

    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=np.pi / 4,
            cycle_time=np.uint(0),
            color=-1,
            max_cycles=1_000_000
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
    params["num_dots"] = 10  # Fewer dots to avoid packing issues
    cycle_time = 12
    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(cycle_time),
            color=-1,
            max_cycles=5
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
            # A dot can also be replaced on frame 0 if its initial 
            # amount_of_cycles was 0
            if not is_end_of_cycle and i > 0:
                assert False, (
                    f"A dot was replaced on frame {i}, which is not the end of "
                    "a cycle."
                )


def test_visibility_with_zero_cycle_time():
    """4. If the cycle_time is 0, all dots are always visible."""
    num_dots = PARAMS["num_dots"]
    direction_props = [
        GroupProperties(
            ratio=0.5, direction=0, cycle_time=np.uint(0), color=-1,
            max_cycles=DEFAULT_MAX_CYCLES
        ),
        GroupProperties(
            ratio=0.5, direction=np.pi, cycle_time=np.uint(0), color=-1,
            max_cycles=DEFAULT_MAX_CYCLES
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
                color=-1,
                max_cycles=DEFAULT_MAX_CYCLES
            )
        ]),
        ("Signal + Noise", [
            GroupProperties(
                ratio=0.5, direction=0, cycle_time=np.uint(10), color=-1,
                max_cycles=DEFAULT_MAX_CYCLES
            ),
            GroupProperties(
                ratio=0.5,
                direction=None,
                cycle_time=np.uint(0),
                color=-1,
                max_cycles=DEFAULT_MAX_CYCLES
            )
        ]),
        ("Two Opposing Signals", [
            GroupProperties(
                ratio=0.5, direction=0, cycle_time=np.uint(10), color=-1,
                max_cycles=DEFAULT_MAX_CYCLES
            ),
            GroupProperties(
                ratio=0.5, direction=np.pi, cycle_time=np.uint(12), color=-1,
                max_cycles=DEFAULT_MAX_CYCLES
            )
        ]),
    ]
)
def test_spatial_distribution_is_unbiased(test_id, direction_props):
    """5. The average mass center is the geometric center (no spatial bias)."""
    params = PARAMS.copy()
    params["duration"] = 5000
    params["motion_velocity"] = 2
    # Ensure max_cycles is valid for these params
    # 500 // 2 // 12 = 20. So 120 is too much. 
    # Let's just set them to 20 for this test.
    for p in direction_props:
        p.max_cycles = 20

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
            color=-1,
            max_cycles=DEFAULT_MAX_CYCLES
        )
    ]

    frames = generate_moving_dots(
        groups_properties=direction_props, **params
    ).frames

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
    }

    cycle_time = 20
    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(cycle_time),
            color=-1,
            max_cycles=5  # Small enough to pass validation
        )
    ]

    frames = generate_moving_dots(
        groups_properties=direction_props, **params
    ).frames

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
        "grid_compression": grid_compression
    }

    direction_props = [
        GroupProperties(
            ratio=0.5,
            direction=0,
            cycle_time=np.uint(2),
            color=-1,
            max_cycles=20
        ),
        GroupProperties(
            ratio=0.5,
            direction=0,
            cycle_time=np.uint(4),
            color=1,
            max_cycles=20
        )
    ]

    frames = generate_moving_dots(
        groups_properties=direction_props, **params
    ).frames

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
        "grid_compression": grid_compression
    }

    direction_props = [
        GroupProperties(
            ratio=1.0,
            direction=0,
            cycle_time=np.uint(0),
            color=-1,
            max_cycles=10
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


def test_different_velocities_can_overlap():
    """
    10. Verify that dots with different velocities can overlap,
    as they use different masks.
    """
    params = {
        "num_dots": 2,
        "dot_radius": 50,
        "stimulus_size_px": 200,
        "duration": 1,
        "motion_velocity": 10,
    }

    # Two groups with different directions (and thus different velocities)
    # Both at the same center if we are "lucky" or if we force it.
    # Actually we can't force it easily via the public API, but we can 
    # check if they overlap in some cases.
    
    # We'll use very large dots in a small area. 
    # With 2 dots of radius 50 in a circle of radius 100, 
    # if they shared a mask, they would be far apart.
    # If they don't share a mask, they MIGHT be placed on top of each other.
    
    # To increase chance of overlap, we can run it many times.
    overlaps_found = False
    for _ in range(100):
        direction_props = [
            GroupProperties(
                ratio=0.5, direction=0, cycle_time=np.uint(0), color=-1,
                max_cycles=10
            ),
            GroupProperties(
                ratio=0.5, direction=np.pi, cycle_time=np.uint(0), color=1,
                max_cycles=10
            )
        ]
        frames = generate_moving_dots(groups_properties=direction_props, **params).frames
        frame = frames[0]
        if len(frame) == 2:
            dist = abs(frame[0].position - frame[1].position)
            if dist < 2 * params["dot_radius"]:
                overlaps_found = True
                break
    
    assert overlaps_found, (
        "Expected dots with different velocities to eventually overlap."
    )


def test_max_cycles_validation():
    """11. Verify that max_cycles validation works."""
    params = {
        "num_dots": 10,
        "dot_radius": 10,
        "stimulus_size_px": 100,
        "duration": 1,
        "motion_velocity": 10,
    }
    # actual_max = 100 // 10 // 2 = 5
    direction_props = [
        GroupProperties(
            ratio=1.0, direction=0, cycle_time=np.uint(2), color=-1,
            max_cycles=6
        )
    ]
    with pytest.raises(ValueError, match="is larger than the actual maximum"):
        generate_moving_dots(groups_properties=direction_props, **params)

    # This should pass
    direction_props[0].max_cycles = 5
    generate_moving_dots(groups_properties=direction_props, **params)


# ==============================================================================
# ## continue_moving_dots Tests (chained simulations)
# ==============================================================================


def _baseline_result(direction=0, cycle_time=0, **overrides):
    """Helper: produce a GenerationResult using PARAMS plus overrides."""
    params = {**PARAMS, **overrides}
    props = [
        GroupProperties(
            ratio=1.0, direction=direction,
            cycle_time=np.uint(cycle_time), color=-1,
            max_cycles=DEFAULT_MAX_CYCLES if cycle_time == 0 else (
                params["stimulus_size_px"] // params["motion_velocity"]
                // cycle_time
            ),
        )
    ]
    return generate_moving_dots(groups_properties=props, **params)


def test_continue_preserves_positions_when_direction_none():
    """direction=None preserves velocity; frame 0 = seed_position + velocity
    for the majority of dots (some may need replacement)."""
    prev = _baseline_result(direction=0)
    seed = list(prev.continuation.last_dot_frame)
    expected = {d.position + d.velocity for d in seed}

    result = continue_moving_dots(
        prev, duration=2,
        modifications=[GroupModification(1.0, None, -1)],
    )
    actual = {d.position for d in result.frames[0]}
    assert len(expected & actual) >= 0.7 * len(seed)


def test_continue_changes_direction_only():
    """A new fixed direction is applied; cycle_time/max_cycles/r unchanged."""
    prev = _baseline_result(direction=0)
    new_direction = np.pi / 2
    expected_velocity = PARAMS["motion_velocity"] * np.exp(
        1j * (new_direction + np.pi / 2)
    )
    seed = list(prev.continuation.last_dot_frame)
    seed_attrs = {
        id(d): (d.cycle_time, d.max_cycles, d.r) for d in seed
    }

    result = continue_moving_dots(
        prev, duration=2,
        modifications=[GroupModification(1.0, new_direction, -1)],
    )
    for dot in result.frames[0]:
        assert np.isclose(dot.velocity, expected_velocity)
    # cycle_time/max_cycles/r unchanged on every dot
    for dot in result.frames[0]:
        # match by (cycle_time, max_cycles, r) since dots are deepcopied
        assert (dot.cycle_time, dot.max_cycles, dot.r) in set(
            seed_attrs.values()
        )


def test_continue_incoherent_randomizes_velocities():
    """direction=INCOHERENT yields varied per-dot velocities at the same speed."""
    prev = _baseline_result(direction=0)
    motion_velocity = prev.continuation.motion_velocity

    result = continue_moving_dots(
        prev, duration=1,
        modifications=[GroupModification(1.0, INCOHERENT, -1)],
    )
    velocities = [d.velocity for d in result.frames[0]]
    # Speeds match motion_velocity
    for v in velocities:
        assert np.isclose(abs(v), motion_velocity)
    # Directions are not all identical
    unique = {round(np.angle(v), 4) for v in velocities}
    assert len(unique) > 1


def test_continue_changes_color():
    """color in the modification is applied to every dot."""
    prev = _baseline_result(direction=0)
    new_color = 1
    result = continue_moving_dots(
        prev, duration=3,
        modifications=[GroupModification(1.0, None, new_color)],
    )
    for frame in result.frames:
        for dot in frame:
            assert dot.visible_color == new_color


def test_continue_phase_continuity():
    """With cycle_time=4, prior_duration multiple of cycle_time, and
    direction=None, the visibility wave is continuous across the seam."""
    cycle_time = 4
    prior_duration = 8  # multiple of cycle_time
    prev = _baseline_result(direction=0, cycle_time=cycle_time,
                            duration=prior_duration)
    cont = continue_moving_dots(
        prev, duration=8,
        modifications=[GroupModification(1.0, None, -1)],
    )
    # Visibility per frame: count visible dots / total at each frame.
    # If phase is continuous, the wave (visible/invisible halves of cycle)
    # holds across the seam.
    all_frames = list(prev.frames) + list(cont.frames)
    expected_total = PARAMS["num_dots"]
    # Frames in the visible half of cycle should have all dots; invisible
    # half should have ~zero.
    for k, frame in enumerate(all_frames):
        if (k % cycle_time) < (cycle_time / 2):
            assert len(frame) > 0.7 * expected_total, (
                f"Frame {k}: visible-half had {len(frame)} dots"
            )
        else:
            assert len(frame) < 0.3 * expected_total, (
                f"Frame {k}: invisible-half had {len(frame)} dots"
            )


def test_continue_ratio_mismatch_raises():
    """Modification ratios must sum to 1.0."""
    prev = _baseline_result(direction=0)
    with pytest.raises(ValueError, match="ratios must sum"):
        continue_moving_dots(
            prev, duration=1,
            modifications=[
                GroupModification(0.4, None, -1),
                GroupModification(0.4, None, 1),
            ],
        )


def test_continue_no_overlap_at_frame_0():
    """No overlap within velocity groups at frame 0 of continuation."""
    prev = _baseline_result(direction=0)
    result = continue_moving_dots(
        prev, duration=1,
        modifications=[
            GroupModification(0.5, 0, -1),
            GroupModification(0.5, np.pi, 1),
        ],
    )
    by_velocity: dict[complex, list] = {}
    for d in result.frames[0]:
        by_velocity.setdefault(d.velocity, []).append(d)
    for group in by_velocity.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                assert (
                    abs(a.position - b.position)
                    >= 2 * PARAMS["dot_radius"] - 1e-6
                )


def test_continue_noop_modification_matches_long_sim():
    """Splitting a long sim into N + continue(M-N) with no-effective-change
    modifications (same direction, same color) produces the same per-frame
    set of dot positions as a single M-frame sim.

    Setup picks huge max_cycles and a large stimulus so no replacement fires
    within M frames; under that condition trajectories are fully determined
    by the initial RNG-seeded placement, and the two runs must match
    frame-by-frame (as multisets, since the continuation shuffles dot order)."""
    direction = np.pi / 4
    color = -1
    motion_velocity = 2
    duration_total = 10
    split_at = 5
    params = dict(
        num_dots=5,
        dot_radius=20,
        stimulus_size_px=2000,
        motion_velocity=motion_velocity,
    )
    huge = 1_000_000
    props = [GroupProperties(
        ratio=1.0, direction=direction, cycle_time=np.uint(0),
        color=color, max_cycles=huge,
    )]

    np.random.seed(42)
    long_result = generate_moving_dots(
        duration=duration_total, groups_properties=props, **params
    )

    np.random.seed(42)
    first_part = generate_moving_dots(
        duration=split_at, groups_properties=props, **params
    )
    second_part = continue_moving_dots(
        first_part,
        duration=duration_total - split_at,
        modifications=[GroupModification(1.0, direction, color)],
    )

    stitched = list(first_part.frames) + list(second_part.frames)
    assert len(stitched) == len(long_result.frames) == duration_total

    def positions(frame):
        return sorted(
            (round(d.position.real, 9), round(d.position.imag, 9))
            for d in frame
        )
    for k, (long_frame, stitched_frame) in enumerate(
        zip(long_result.frames, stitched)
    ):
        assert positions(long_frame) == positions(stitched_frame), (
            f"Frame {k} positions differ between long and stitched sims"
        )


def test_continue_outward_velocity_triggers_replacement():
    """A dot at the left edge with a new leftward direction gets replaced
    via the standard replacement path rather than carried forward."""
    stimulus_size_px = 500
    dot_radius = 20
    stimulus_radius = stimulus_size_px / 2.0
    outer = stimulus_radius - dot_radius
    center = stimulus_radius * (1 + 1j)
    edge_pos = center + complex(-(outer - 1), 0)

    seed_dot = Dot(
        position=edge_pos, r=dot_radius,
        velocity=complex(2, 0),  # was moving right
        death_frame=10, is_coherent=True, is_visible=True,
        visible_color=-1, cycle_time=0,
        max_cycles=DEFAULT_MAX_CYCLES,
        placement_radius=outer,
    )
    prev = GenerationResult(
        frames=[[seed_dot]], markers=[],
        continuation=ContinuationContext(
            last_dot_frame=[seed_dot],
            motion_velocity=2,
            dot_radius=dot_radius,
            stimulus_size_px=stimulus_size_px,
            grid_compression=10.0,
            duration=1,
            cycle_times=(0,),
        ),
    )
    # New direction points left (outward) → -π/2 angle gives velocity
    # along -x axis (since velocity = motion_velocity * exp(1j * (dir + π/2))).
    result = continue_moving_dots(
        prev, duration=1,
        modifications=[GroupModification(1.0, -np.pi / 2, -1)],
    )
    placed = result.frames[0][0].position
    assert abs(placed - center) <= outer + 1e-6
    # Should NOT be a simple outward step from edge_pos.
    assert abs(placed - (edge_pos + complex(-2, 0))) > 1e-6


def test_continuation_is_none_when_duration_breaks_phase():
    """duration not a multiple of cycle_time → continuation is None."""
    cycle_time = 4
    actual_max = (
        PARAMS["stimulus_size_px"] // PARAMS["motion_velocity"] // cycle_time
    )
    props = [GroupProperties(
        ratio=1.0, direction=0,
        cycle_time=np.uint(cycle_time), color=-1, max_cycles=actual_max,
    )]
    result = generate_moving_dots(
        groups_properties=props, **{**PARAMS, "duration": 7},
    )
    assert result.continuation is None
    # Frames and markers are still produced normally.
    assert len(result.frames) == 7


def test_continue_raises_when_continuation_is_none():
    """continue_moving_dots must raise if previous.continuation is None."""
    cycle_time = 4
    actual_max = (
        PARAMS["stimulus_size_px"] // PARAMS["motion_velocity"] // cycle_time
    )
    props = [GroupProperties(
        ratio=1.0, direction=0,
        cycle_time=np.uint(cycle_time), color=-1, max_cycles=actual_max,
    )]
    result = generate_moving_dots(
        groups_properties=props, **{**PARAMS, "duration": 7},
    )
    with pytest.raises(ValueError, match="no continuation context"):
        continue_moving_dots(
            result, duration=4,
            modifications=[GroupModification(1.0, None, -1)],
        )


def test_continuation_context_validates_at_construction():
    """Constructing ContinuationContext with a misaligned duration raises."""
    seed_dot = Dot(
        position=0j, r=20, velocity=0j, death_frame=0,
        is_coherent=True, is_visible=True, visible_color=-1,
        cycle_time=4, max_cycles=1, placement_radius=230,
    )
    with pytest.raises(ValueError, match="multiple of"):
        ContinuationContext(
            last_dot_frame=[seed_dot],
            motion_velocity=2, dot_radius=20,
            stimulus_size_px=500, grid_compression=10.0,
            duration=7, cycle_times=(4,),
        )
    # cycle_time=0 → no constraint, any duration ok.
    ContinuationContext(
        last_dot_frame=[seed_dot],
        motion_velocity=2, dot_radius=20,
        stimulus_size_px=500, grid_compression=10.0,
        duration=7, cycle_times=(0,),
    )


