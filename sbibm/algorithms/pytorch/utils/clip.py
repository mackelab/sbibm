from __future__ import annotations


def clip_int(value, minimum, maximum):
    value = int(value)
    minimum = int(minimum)
    maximum = int(maximum)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
