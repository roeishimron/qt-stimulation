from dataclasses import dataclass
from numpy import sin, pi, linspace, uint8, int16, float64, int64, tile, full, ones, square, mgrid, array, argwhere, logical_not, log, diff, exp, meshgrid, inf, arange, zeros
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

    RANGE = 23

    x, y = meshgrid(linspace(-RANGE, RANGE, size),
                    linspace(-RANGE, RANGE, size))

    # Calculating Gaussian filter

    return exp(-(x**2 + y**2)/sigma)


def create_gabor_values(figure_size, frequency=1, offset=0, contrast=1, horizontal=False, step=False, raidal_easing=inf) -> NDArray:
    assert figure_size >= 2*frequency

    sin_flips = (frequency*figure_size)
    virtual_size = sin_flips * 2 * pi

    x_range = linspace(offset, virtual_size + offset, figure_size)
    sinsusoid = sin(x_range)*contrast

    if step:
        sinsusoid = (sinsusoid > 0)*2-1

    frame = tile(sinsusoid, (figure_size, 1))

    if not horizontal:
        frame = frame.transpose()

    frame = frame * gaussian(figure_size, raidal_easing)

    return frame


def generate_sin(figure_size, frequency=1, offset=0, contrast=1, horizontal=False, step=False, raidal_easing=inf) -> AppliablePixmap:
    return array_into_pixmap(
        create_gabor_values(
            figure_size, frequency, offset, contrast, horizontal, step, raidal_easing))

# assuming array of values between -1 and 1
def array_into_pixmap(arr: NDArray) -> AppliablePixmap:
    mormalized_to_pixels = (((arr * 255) + 255)/2)
    mormalized_to_pixels = array(mormalized_to_pixels, dtype=uint8)

    return AppliablePixmap(QPixmap.fromImage(QImage(mormalized_to_pixels, arr.shape[1], arr.shape[0], arr.shape[1], QImage.Format.Format_Grayscale8)))


def generate_grey(figure_size: int) -> AppliablePixmap:
    return generate_sin(figure_size, 0, 0, 0)

# left and right are between -1 and 1
def place_in_figure(figure_size: Tuple[int,int], left: NDArray, right: NDArray) -> AppliablePixmap:
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


def generate_noise(width: int, height: int) -> AppliablePixmap:
    bools = rand(height, width) > 0.5
    return array_into_pixmap(bools * 2 - 1)


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

# Note: Should have been split into several functions. It's avoided to save preformance.


def fill_with_white_dots(figure_size: int, amount_of_dots: int, dot_size: int) -> AppliablePixmap:
    fillers = ones(shape=(amount_of_dots, dot_size, dot_size))
    return fill_with_dots(figure_size, fillers)

# returns is_horizontal and (x,y)


def rec_at(position: int, width: int, height: int) -> Tuple[bool, Tuple[int, int]]:
    vecs = [array([0,height]), array([width,0]), array([0,-height]), array([-width,0])]
    assert position <= sum(abs(array(vecs).flatten()))
    current_position = (0,0)
    for v in vecs:
        length = sum(abs(v))
        if position - length < 0:
            exact = current_position + v/length*(position)
            return (v[0]!=0, (exact[0], exact[1]))
        position -= length
        current_position += v


def gabors_around_rec(width: int, height: int, amount_of_dots: int,
                       offset: int, dot_size: int, gabor_freq: int) -> List[NDArray]:
    length = 2*(width+height)
    rects = [rec_at(i, width, height) for i in range(0, length, int(length/amount_of_dots))]
    return [
        Dot(dot_size/2, array([x, y]) + offset, create_gabor_values(dot_size,
            horizontal=horizontal, frequency=gabor_freq, raidal_easing=150))
        for (horizontal, (x, y)) in rects]


def fill_with_dots(figure_size: int, dots_fill: List[NDArray], priority_dots: List[NDArray] = []) -> AppliablePixmap:
    # there must be enough room for all the dots
    amount_of_dots = len(dots_fill) + len(priority_dots)
    dot_size = dots_fill[0].shape[0]

    assert dots_fill[0].shape[0] == dots_fill[0].shape[1]
    assert figure_size**2 >= amount_of_dots * (pi*(dot_size)**2)

    available_positions = full((figure_size, figure_size), True)
    canvas = full((figure_size, figure_size), 0, float64)

    available_positions[:, :int(dot_size/2)] = False
    available_positions[:, -int(dot_size/2):] = False
    available_positions[:int(dot_size/2), :] = False
    available_positions[-int(dot_size/2):, :] = False

    filler_xs, filler_ys = mgrid[:dot_size, :dot_size]
    filler_mask = argwhere(square(filler_xs - dot_size/2) +
                           square(filler_ys - dot_size/2) <= square(dot_size/2))

    xs, ys = mgrid[:figure_size, :figure_size]
    complete_requirement = priority_dots + \
        [Dot(dot_size/2, array([]), fill) for fill in dots_fill]
    for dot in complete_requirement:
        if dot.position.shape[0] == 0:
            remaining_position_indices = argwhere(available_positions)
            next_center = remaining_position_indices[choice(
                len(remaining_position_indices))]

            dot.position = array([next_center[0], next_center[1]])

        shifted_mask = filler_mask + \
            array([dot.position[0] - dot.r, dot.position[1] - dot.r], dtype=int64)
        canvas[shifted_mask[:, 0], shifted_mask[:, 1]
               ] = dot.fill[filler_mask[:, 0], filler_mask[:, 1]]

        circle = square(xs - dot.position[0]) + square(ys - dot.position[1])
        available_positions &= logical_not(circle <= square(dot.r*2))

    return canvas

def dots_image(figure_size: int, dots_fill: List[NDArray], priority_dots: List[NDArray] = []) -> AppliablePixmap:
    return array_into_pixmap(
        fill_with_dots(figure_size, dots_fill, priority_dots)
    )