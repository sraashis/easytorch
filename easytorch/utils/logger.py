import math as _math


def error(msg, debug=True):
    if debug:
        print(f"[#### Error!  ####]: {msg}")


def warn(msg, debug=True):
    if debug:
        print(f"[---  Warning! ---]: {msg}")


def info(msg, debug=True):
    if debug: print(f"{msg}")


def success(msg, debug=True):
    if debug:
        print(f"[***  Success! ***]: {msg}")


def lazy_debug(x, add=1):
    return x % int(_math.log(x + 1) + add) == 0
