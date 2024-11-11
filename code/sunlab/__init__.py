from .common import *

font = {"family": "DejaVu Sans", "weight": "regular", "size": 20}
import matplotlib

matplotlib.rc("font", **font)


def set_font(ptsize=10):
    font = {"family": "DejaVu Sans", "weight": "regular", "size": ptsize}
    import matplotlib

    matplotlib.rc("font", **font)


def set_font_l(ptsize=20):
    set_font(ptsize)
