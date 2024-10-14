from numpy import sin, pi, linspace, uint8, tile
from PySide6.QtGui import QPixmap, QImage, QColor
from animator import AppliablePixmap


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

def generate_solid_color(figure_size: int, h: int, s: int=255, v: int=255):
    image = QImage(figure_size, figure_size, QImage.Format.Format_RGB16)
    color = QColor()
    color.setHsv(h,s,v)
    image.fill(color)
    return AppliablePixmap(QPixmap.fromImage(image))
    