from dataclasses import dataclass
from numpy import cos, sin, pi, linspace, sqrt, uint8, int16, float64, int64, tile, full, ones, square, mgrid, array, argwhere, logical_not, log, diff, exp, meshgrid, inf, arange, zeros, where, all
from numpy.random import choice, rand
from numpy.typing import NDArray
from PySide6.QtGui import QPixmap, QImage, QColor
from animator import AppliablePixmap
import sys
from PySide6.QtGui import QPixmap, QImage
from animator import AppliablePixmap
from itertools import chain
from typing import List, Generator, Any, Iterable, Tuple
from random import shuffle, randint


def gaussian(size: int, sigma):

    RANGE = 70

    x, y = meshgrid(linspace(-RANGE, RANGE, size),
                    linspace(-RANGE, RANGE, size))

    # Calculating Gaussian filter

    return exp(-(x**2 + y**2)/sigma)


def _create_rotated_sin_frame(figure_size, frequency, offset, contrast, rotation) -> NDArray:
    sin_flips = frequency
    virtual_size = sin_flips * 2 * pi
    assert figure_size % 2 == 0

    new_x_axis = array([sin(rotation), cos(rotation)])
    orthogonal_values = zeros((figure_size, figure_size))
    for x in range(int(-figure_size/2), int(figure_size/2)):
        for y in range(int(-figure_size/2), int(figure_size/2)):
            orthogonal_values[int(x+figure_size/2), int(y+figure_size/2)
                              ] = sum(array([x, y]) * new_x_axis) 

    # normalize so that the max length arrives at the right amount of flips
    orthogonal_values = orthogonal_values * virtual_size / figure_size
    return sin(orthogonal_values+offset)*contrast


def _create_unrotated_sin_frame(figure_size, frequency, offset, contrast, horizontal: bool) -> NDArray:
    sin_flips = (frequency*figure_size)
    virtual_size = sin_flips * 2 * pi

    x_range = linspace(offset, virtual_size + offset, figure_size)
    sinsusoid = sin(x_range)*contrast

    frame = tile(sinsusoid, (figure_size, 1))

    if not horizontal:
        frame = frame.transpose()
    return frame

# currently, the frequency is NOT accurate due to the rotation support. It IS accurate for 0,90deg


def create_gabor_values(figure_size, frequency=1, offset=0, contrast=1,
                        step=False, raidal_easing=inf, rotation=0) -> NDArray:
    assert figure_size >= 2*frequency
    frame = None
    if rotation % (pi/2) == 0:
        frame = _create_unrotated_sin_frame(
            figure_size, frequency, offset, contrast, rotation % pi/2 == 0)
    else:
        frame = _create_rotated_sin_frame(
            figure_size, frequency, offset, contrast, rotation)

    if step:
        frame = (frame > 0)*2-1

    frame = frame * gaussian(figure_size, raidal_easing)

    return frame


def generate_sin(figure_size, frequency=1, offset=0, contrast=1, step=False, raidal_easing=inf, rotation=0) -> AppliablePixmap:
    return array_into_pixmap(
        create_gabor_values(
            figure_size, frequency, offset, contrast, step, raidal_easing, rotation))

# assuming array of values between -1 and 1


def array_into_pixmap(arr: NDArray) -> AppliablePixmap:
    mormalized_to_pixels = (((arr * 255) + 256)/2)
    mormalized_to_pixels = array(mormalized_to_pixels, dtype=uint8)

    return AppliablePixmap(QPixmap.fromImage(QImage(mormalized_to_pixels, arr.shape[1], arr.shape[0], arr.shape[1], QImage.Format.Format_Grayscale8)))


def generate_grey(figure_size: int) -> AppliablePixmap:
    return generate_sin(figure_size, 0, 0, 0)

# left and right "pixel" values are between -1 and 1


def place_in_figure(figure_size: Tuple[int, int], left: NDArray, right: NDArray) -> AppliablePixmap:
    assert left.shape[1] + right.shape[1] <= figure_size[1]
    assert max(left.shape[0], right.shape[0]) <= figure_size[0]

    # Assuming both middle
    grid = zeros((figure_size[0], figure_size[1]), dtype=float64)
    left_center = array([figure_size[0]/2, figure_size[1]/4], dtype=int)
    grid[left_center[0] - int(left.shape[0]/2):left_center[0] + int(left.shape[0]/2),
         left_center[1] - int(left.shape[1]/2):left_center[1] + int(left.shape[1]/2)] = left
    right_center = array([figure_size[0]/2, figure_size[1]/4*3], dtype=int)
    grid[right_center[0] - int(right.shape[0]/2):right_center[0] + int(right.shape[0]/2),
         right_center[1] - int(right.shape[1]/2):right_center[1] + int(right.shape[1]/2)] = right

    return array_into_pixmap(grid)


def generate_solid_color(figure_size: int, h: int, s: int = 255, v: int = 255):
    image = QImage(figure_size, figure_size, QImage.Format.Format_RGB16)
    color = QColor()
    color.setHsv(h, s, v)
    image.fill(color)
    return AppliablePixmap(QPixmap.fromImage(image))


def generate_noise(width: int, height: int, kernel_size: int = 1) -> AppliablePixmap:
    assert width % kernel_size == 0 and height % kernel_size == 0
    virtual_height = int(height / kernel_size)
    virtual_width = int(width / kernel_size)

    bools = rand(virtual_height, virtual_width) > 0.5

    screen = zeros((height, width))

    for row in range(virtual_height):
        for col in range(virtual_width):
            screen[row*kernel_size:(row+1)*kernel_size, col *
                   kernel_size:(col+1)*kernel_size] = bools[row, col]

    return array_into_pixmap(screen * 2 - 1)


def inflate_randomley(source: List[Any], factor: int) -> List[Any]:
    def inflate() -> Generator[List[Any], None, None]:
        for _ in range(factor):
            current = list(source)
            shuffle(current)
            yield current

    return list(chain.from_iterable(inflate()))

# taking to f(r(x)) when r(x) is the exponent and f(x) = sin(pi*F*x) for frequency F


def generate_increasing_durations(alleged_frequency: int) -> List[int]:
    TRIAL_DURATION = 50
    amount_of_stimuli = TRIAL_DURATION * alleged_frequency
    ks = arange(amount_of_stimuli)
    peaks = 2*ks/alleged_frequency

    SCALE = 20
    A = 9
    modified_peaks = SCALE*log(peaks/A+1)
    # should be the same at the trial end
    duration_in_s = diff(modified_peaks)
    print(f"from {1/duration_in_s[0]} up to {1/duration_in_s[-1]}")

    return list(duration_in_s * 1000)


@dataclass
class Dot:
    r: int
    position: NDArray
    fill: NDArray

# returns is_horizontal and (x,y)


def rec_at(position: int, width: int, height: int) -> Tuple[float64, Tuple[int, int]]:
    vecs = [array([0, height]), array([width, 0]),
            array([0, -height]), array([-width, 0])]
    assert position <= sum(abs(array(vecs).flatten()))
    current_position = (0, 0)
    for v in vecs:
        length = sum(abs(v))
        if position - length < 0:
            exact = current_position + v/length*(position)
            if v[0] == 0:
                return (pi/2, (exact[0], exact[1]))
            return (0, (exact[0], exact[1]))

        position -= length
        current_position += v


def circle_at(center: Tuple[int, int], radius: int, angle_from_0: float64) -> Tuple[float64, Tuple[int, int]]:

    coordinates = array(
        array(center) + array([cos(angle_from_0), sin(angle_from_0)])*radius, dtype=int)
    ortho = pi/2 - angle_from_0
    return (ortho, (coordinates[0], coordinates[1]))


def gabors_around_rec(width: int, height: int, amount_of_dots: int,
                      offset: int, dot_size: int, gabor_freq: int,
                      raidal_easing: int) -> List[Dot]:
    length = 2*(width+height)
    rects = [rec_at(i, width, height)
             for i in range(0, length, int(length/amount_of_dots))]
    return [
        Dot(dot_size/2, array([x, y]) + offset, create_gabor_values(dot_size,
            rotation=rotation, frequency=gabor_freq, raidal_easing=raidal_easing))
        for (rotation, (x, y)) in rects]


def gabors_around_circle(center: Tuple[int, int], radius: int, amount_of_dots: int,
                         dot_size: int, gabor_freq: int, radial_easing: int, flip_one=False,
                         offset=0, fill_reduction=0) -> List[Dot]:
    angles = linspace(offset, pi*2 + offset, amount_of_dots)
    properties = [circle_at(center, radius, angle) for angle in angles]

    flip_index = randint(0, len(angles)-1) if flip_one else -1

    return [Dot(dot_size/2, array(position),
                create_gabor_values(dot_size, gabor_freq, raidal_easing=radial_easing,
                                    rotation=rotation+pi/2 if i == flip_index else rotation)-fill_reduction)
            for (i, (rotation, position)) in enumerate(properties)]


def get_false_margins(radius: int, figure_size: int):
    available_positions = full((figure_size, figure_size), True)

    available_positions[:, :int(radius)] = False
    available_positions[:, -int(radius):] = False
    available_positions[:int(radius), :] = False
    available_positions[-int(radius):, :] = False

    return available_positions

# Set only dots without position
def place_dots_in_square(figure_size:int, dots: List[Dot], minimum_distance_factor: float=1) -> List[Dot]:
    
    available_positions = full((figure_size, figure_size), True)
    xs, ys = mgrid[:figure_size, :figure_size]

    for dot in dots:
        if dot.position.shape[0] == 0:
            remaining_position_indices = argwhere(
                available_positions & get_false_margins(dot.r, figure_size))

            dot.position = remaining_position_indices[choice(
                len(remaining_position_indices))]

        circle = square(xs - dot.position[0]) + square(ys - dot.position[1])
        available_positions &= logical_not(circle <= square(dot.r*2*minimum_distance_factor))
    return dots
        

# there must be enough room for all the dots
# Priority dots are allowed to be without position
def fill_with_dots(figure_size: int,
                   dots_fill: List[NDArray],
                   priority_dots: List[Dot] = [],
                   backdround_value: float = 0,
                   minimum_distance_factor: float=1) -> NDArray:

    canvas = full((figure_size, figure_size), backdround_value, float64)

    complete_requirement = priority_dots + \
        [Dot(int(fill.shape[0]/2), array([]), fill) for fill in dots_fill]
    
    complete_requirement = place_dots_in_square(figure_size, complete_requirement, minimum_distance_factor)
    
    for dot in complete_requirement:
        filler_xs, filler_ys = mgrid[:dot.r*2, :dot.r*2]
        filler_mask = argwhere(square(filler_xs - dot.r) +
                               square(filler_ys - dot.r) < square(dot.r))

        shifted_mask = filler_mask + \
            array([dot.position[0] - dot.r, dot.position[1] - dot.r], dtype=int64)
        canvas[shifted_mask[:, 0], shifted_mask[:, 1]
               ] = dot.fill[filler_mask[:, 0], filler_mask[:, 1]]

    return canvas
