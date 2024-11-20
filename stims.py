from dataclasses import dataclass
from numpy import sin, pi, linspace, uint8, tile, full, sqrt, square, mgrid, array
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


def inflate_randomley(source: List[Any], factor: int) -> Iterable[Any]:
    def inflate() -> Generator[List[Any], None, None]:
        for _ in range(factor):
            current = list(source)
            shuffle(current)
            yield current

    return chain.from_iterable(inflate())


@dataclass
class Dot:
    x: int
    y: int
    r: int


def is_dot_valid(r: int, x_center: int, y_center: int, existing_dots: List[Dot]) -> bool:
    for d in existing_dots:
        if sqrt(square(x_center - d.x) + square(y_center - d.y)) < r + d.r:
            return False
    return True

def dots_into_image(dots: List[Dot], figure_size: int) -> NDArray:
    positions = full((figure_size, figure_size), False)
    xs, ys = mgrid[:figure_size, :figure_size]
    for d in dots:
        circle = square(xs - d.x) + square(ys - d.y)
        mask = circle <= d.r
        positions = positions | mask
    image = array(positions, dtype=uint8) * 127 + 127
    return image


def fill_with_dots(figure_size: int, amount_of_dots: int, dot_size: int) -> AppliablePixmap:
    # there must be enough room for all the dots
    assert figure_size**2 >= amount_of_dots * ((dot_size*2)**2)

    dots = []
    while amount_of_dots > 0:
        (x, y) = (randint(dot_size, figure_size - dot_size), randint(dot_size, figure_size - dot_size))
        if is_dot_valid(dot_size, x, y, dots):
            dots.append(Dot(x, y, dot_size))
            amount_of_dots -= 1

    image = dots_into_image(dots, figure_size)
    return AppliablePixmap(QPixmap.fromImage(QImage(image, figure_size, figure_size, figure_size, QImage.Format.Format_Grayscale8)))

