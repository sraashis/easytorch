import math as _math


def error(msg):
    print(f"####   [Error!]   ####: {msg}")


def warn(msg):
    print(f"---  [Warning!]  ---: {msg}")


def info(msg):
    print(f"{msg}")


def success(msg):
    print(f"***  [Success!] ***: {msg}")


def lazy_debug(x, add=1):
    return x % int(_math.log(x + 1) + add) == 0
