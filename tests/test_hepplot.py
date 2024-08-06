"""
statistics-notebooks pytests
"""

import math

import hepplot as hep


def test_hist1d():
    bins = list(range(6))
    y = [ [9,8,6,4,1], [12,10,7,2,2], [6,10,5,8,2] ]
    data = [26, 30, 19, 12, 6]
    labels = ["A","B","C"]
    ytotal  = [sum(i) for i in zip(*y)]

    yerr    = [
        0.2*math.sqrt(_y) for _y in ytotal
    ]
    yerrs   = [
        [0.2*math.sqrt(_y) for _y in y_i] for y_i in y
    ]

    fig, axes = hep.plot.hist1d(bins, y, yerrs=yerrs, data=data, labels=labels, ratio=True)
    assert fig

