from dataclasses import dataclass
from numpy import sin, pi, linspace, uint8, tile, full, sqrt, square, mgrid, array, argwhere, logical_not, log, diff, exp, meshgrid, inf, arange
from numpy.random import choice
from numpy.typing import NDArray
from PySide6.QtGui import QPixmap, QImage, QColor
from animator import AppliablePixmap

import sys
from PySide6.QtGui import QPixmap, QImage
from animator import AppliablePixmap
from itertools import chain
from typing import List, Generator, Any, Iterable
from random import shuffle, randint



def gaussian(size: int, sigma):
    
    RANGE = 23

    x, y = meshgrid(linspace(-RANGE, RANGE, size),
                    linspace(-RANGE, RANGE, size))

    # Calculating Gaussian filter
    return exp(-(x**2 + y**2)/sigma)


def generate_sin(figure_size, frequency=1, offset=0, contrast=1, horizontal=False, step=False, raidal_easing=inf) -> AppliablePixmap:

    assert figure_size >= 2*frequency

    sin_flips = (frequency*figure_size)
    virtual_size = sin_flips * 2 * pi

    x_range = linspace(offset, virtual_size + offset, figure_size)
    sinsusoid = sin(x_range)
    if step:
        sinsusoid = (sinsusoid > 0)*2-1
    frame = tile(sinsusoid, (figure_size, 1))

    if horizontal:
        frame = frame.transpose()   

    shaded = frame * gaussian(figure_size, raidal_easing)

    mormalized_to_pixels = (((shaded * 255) * contrast + 255)/2)
    mormalized_to_pixels = array(mormalized_to_pixels, dtype=uint8)

    return AppliablePixmap(QPixmap.fromImage(QImage(mormalized_to_pixels, figure_size, figure_size, figure_size, QImage.Format.Format_Grayscale8)))


def generate_grey(figure_size: int) -> AppliablePixmap:
    return generate_sin(figure_size, 0, 0, 0)


def generate_solid_color(figure_size: int, h: int, s: int = 255, v: int = 255):
    image = QImage(figure_size, figure_size, QImage.Format.Format_RGB16)
    color = QColor()
    color.setHsv(h, s, v)
    image.fill(color)
    return AppliablePixmap(QPixmap.fromImage(image))


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
    #should be the same at the trial end
    duration_in_s = diff(modified_peaks)
    print(f"from {1/duration_in_s[0]} up to {1/duration_in_s[-1]}")

    return list(duration_in_s * 1000)



@dataclass
class Dot:
    x: int
    y: int
    r: int

# Note: Should have been split into several functions. It's avoided to save preformance.
def fill_with_dots(figure_size: int, amount_of_dots: int, dot_size: int) -> AppliablePixmap:
    # there must be enough room for all the dots
    assert figure_size**2 >= amount_of_dots * ((dot_size)**2)

    available_positions = full((figure_size, figure_size), True)
    canvas = full((figure_size, figure_size), False)

    available_positions[:, :dot_size] = False
    available_positions[:, -dot_size:] = False
    available_positions[:dot_size, :] = False
    available_positions[-dot_size:, :] = False

    xs, ys = mgrid[:figure_size, :figure_size]

    for _ in range(amount_of_dots):
        remaining_position_indices = argwhere(available_positions)
        next_center = remaining_position_indices[choice(
            len(remaining_position_indices))]
        
        dot = Dot(next_center[0], next_center[1], dot_size)
        circle = square(xs - dot.x) + square(ys - dot.y)

        canvas |= (circle <= square(dot.r))
        available_positions &= logical_not(circle <= square(dot.r*2))

    image = array(canvas, dtype=uint8) * 127 + 127
    return AppliablePixmap(QPixmap.fromImage(QImage(image, figure_size, figure_size, figure_size, QImage.Format.Format_Grayscale8)))
