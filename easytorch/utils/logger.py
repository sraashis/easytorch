import math as _math


def error(msg, debug=True):
    if debug:
        print(f"[ERROR]! {msg}")


def warn(msg, debug=True):
    if debug:
        print(f"[WARNING]! {msg}")


def info(msg, debug=True):
    if debug: print(f"{msg}")


def success(msg, debug=True):
    if debug:
        print(f"[SUCCESS]! {msg}")


def lazy_debug(x, add=0):
    _scale = int(_math.log(max(x, 1)) * _math.log(max(add, 1)))
    return x % (_scale + 1) == 0
