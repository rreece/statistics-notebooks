"""
hepplot/stat.py

See:
https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
https://en.wikipedia.org/wiki/Chi-square_distribution#Asymptotic_properties
https://www.johndcook.com/blog/wilson_hilferty/
"""


import math


def poisson_error_up(data):
    y1 = data + 1.0
    d = 1.0 - 1.0/(9.0*y1) + 1.0/(3*math.sqrt(y1))
    return y1*d*d*d-data


def poisson_error_down(data):
    y = data
    if y == 0.0: return 0.0
    d = 1.0 - 1.0/(9.0*y) - 1.0/(3.0*math.sqrt(y))
    return data-y*d*d*d
