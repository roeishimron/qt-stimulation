import numpy as np
import copy
from dataclasses import dataclass
from typing import List

# Define the Dot data structure
@dataclass
class Dot:
    """Represents a single dot in the stimulus."""
    position: complex
    r: int
    velocity: complex
    age: int
    lifetime: int
    is_coherent: bool
    color: int

    @property
    def x(self) -> float:
        """The x-coordinate of the dot."""
        return self.position.real

    @property
    def y(self) -> float:
        """The y-coordinate of the dot."""
        return self.position.imag

# ==============================================================================
# ## Helper Functions
# ==============================================================================

def _get_new_lifetime(mean_lifetime: int) -> int:
    """Calculates a dot's lifetime from an exponential distribution."""
    return max(1, int(np.random.exponential(scale=mean_lifetime)))

def _find_valid_position(
    effective_radius: float,
    dot_radius: int,
    existing_dots: List[Dot],
    velocity: complex,
    center: complex,
    max_tries: int = 1000
) -> complex:
    """
    Finds a valid, non-overlapping position for a dot using complex arithmetic.
    """
    effective_radius_sq = effective_radius**2
    c2 = -velocity

    for _ in range(max_tries):
        rel_x = np.random.uniform(-effective_radius, effective_radius)
        rel_y = np.random.uniform(-effective_radius, effective_radius)
        rel_pos = rel_x + 1j * rel_y

        if not (abs(rel_pos)**2 < effective_radius_sq and abs(rel_pos - c2)**2 < effective_radius_sq):
            continue

        candidate_pos = center + rel_pos
        is_overlapping = False
        for dot in existing_dots:
            if abs(candidate_pos - dot.position)**2 < (2 * dot_radius)**2:
                is_overlapping = True
                break
        
        if not is_overlapping:
            return candidate_pos

    raise RuntimeError("Could not find a valid non-overlapping position.")

def _create_direction_markers(
    direction: float, # In RADIANS
    dot_radius: int,
    center: complex,
    placement_radius: float
) -> List[Dot]:
    """
    Creates four static dots as markers using a vectorized numpy approach.
    """
    # CHANGED: Refactored to use linspace and a single rotation
    
    # 1. Define the four orthogonal base angles (0, 90, 180, 270 degrees)
    base_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)

    # 2. Create the initial four position vectors, scaled by the radius
    relative_positions = placement_radius * np.exp(1j * base_angles)

    # 3. Calculate the single rotation vector needed to align with the coherent direction
    #    (The pi/2 offset maintains the visual convention)
    rotation_angle = direction + np.pi / 2
    rotation_vector = np.exp(1j * rotation_angle)

    # 4. Apply the rotation to all four vectors at once
    rotated_relative_positions = relative_positions * rotation_vector
    
    markers = []
    for rel_pos in rotated_relative_positions:
        marker_dot = Dot(
            position=center + rel_pos,
            r=dot_radius,
            velocity=0 + 0j,
            age=0,
            lifetime=float('inf'),
            is_coherent=False,
            color=1
        )
        markers.append(marker_dot)
        
    return markers

# ==============================================================================
# ## Main Function
# ==============================================================================

def generate_moving_dots(
    num_dots: int,
    dot_radius: int,
    coherence_level: float,
    stimulus_size_px: int,
    duration: int,
    coherent_direction: float, # In RADIANS
    motion_velocity: int,
    mean_lifetime: int
) -> List[List[Dot]]:
    """
    Generates frames of moving dots for a Random Dot Kinematogram (RDK).
    """
    if not (0.0 <= coherence_level <= 1.0):
        raise ValueError("coherence_level must be between 0.0 and 1.0.")
    
    # Calculate geometric properties once
    stimulus_radius = stimulus_size_px / 2.0
    center = (stimulus_size_px / 2.0) + 1j * (stimulus_size_px / 2.0)
    effective_radius = stimulus_radius - (3 * dot_radius)
    marker_placement_radius = stimulus_radius - dot_radius
    
    if effective_radius <= 0:
        raise ValueError("Stimulus size is too small for the given dot radius and marker margin.")

    rotated_direction_rad = coherent_direction + np.pi / 2
    coherent_velocity = motion_velocity * np.exp(1j * rotated_direction_rad)

    dots: List[Dot] = []
    num_coherent_dots = int(num_dots * coherence_level)
    
    for i in range(num_dots):
        is_coherent = i < num_coherent_dots
        velocity = coherent_velocity if is_coherent else motion_velocity * np.exp(1j * (np.random.rand() * 2 * np.pi))
        lifetime = _get_new_lifetime(mean_lifetime)
        
        position = _find_valid_position(
            effective_radius, dot_radius, dots, velocity, center
        )
        
        dots.append(
            Dot(
                position=position, r=dot_radius, velocity=velocity,
                age=np.random.randint(0, lifetime),
                lifetime=lifetime, is_coherent=is_coherent,
                color=-1
            )
        )

    moving_dot_frames = []
    for _ in range(duration):
        moving_dot_frames.append(copy.deepcopy(dots))
        
        for dot in dots:
            dot.position += dot.velocity
            dot.age += 1
            
            needs_new_position = False
            
            if dot.age > dot.lifetime:
                dot.age = 0
                dot.lifetime = _get_new_lifetime(mean_lifetime)
                if not dot.is_coherent:
                    dot.velocity = motion_velocity * np.exp(1j * (np.random.rand() * 2 * np.pi))
                needs_new_position = True
            
            elif abs(dot.position - center) > effective_radius:
                needs_new_position = True

            if needs_new_position:
                other_dots = [d for d in dots if d is not dot]
                dot.position = _find_valid_position(
                    effective_radius, dot.r, other_dots, dot.velocity, center
                )

    # After the simulation, create the markers using the pre-calculated values
    markers = _create_direction_markers(
        coherent_direction, dot_radius, center, marker_placement_radius
    )
    
    final_frames = [frame + markers for frame in moving_dot_frames]
                
    return final_frames