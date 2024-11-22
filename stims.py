from dataclasses import dataclass
from numpy import sin, pi, linspace, uint8, tile, full, sqrt, square, mgrid, array, argwhere, logical_not
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


def generate_sin(figure_size, frequency=1, offset=0, contrast=1) -> AppliablePixmap:

    assert figure_size >= 2*frequency

    sin_flips = (frequency*figure_size)
    virtual_size = sin_flips * 2 * pi

    x_range = linspace(offset, virtual_size + offset, figure_size)
    sinsusoid = sin(x_range)

    mormalized_to_pixels = (((sinsusoid * 255) * contrast + 255)/2)

    image_line = mormalized_to_pixels.astype(uint8)
    raw_array = tile(image_line, (figure_size, 1))

    return AppliablePixmap(QPixmap.fromImage(QImage(raw_array, figure_size, figure_size, figure_size, QImage.Format.Format_Grayscale8)))


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
