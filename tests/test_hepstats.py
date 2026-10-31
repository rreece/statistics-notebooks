"""
statistics-notebooks pytests
"""

import math
import numpy as np

import hepplot as hep


def test_hypo_test_mu_scan():
    """
    Hypothesis test scan for a single signal
    """
    y = [ [9,8,6,4,1], [12,10,7,2,2], [6,10,5,8,2] ]
    data = [26, 30, 19, 12, 6]
    ytotal  = [sum(i) for i in zip(*y)]

    yerr    = [
        0.2*math.sqrt(_y) for _y in ytotal
    ]

    signal_data = [0, 0, 1, 6, 10]
    mu_bounds_excl = (0, 4)
    mu_step_excl = 0.05
    test_size = 0.05

    first_signal_pdf = hep.stats.make_pdf(bkg_data=ytotal, bkg_uncerts=yerr, signal_data=signal_data)
    cls_obs, cls_exp, test_mus = hep.stats.hypo_test_mu_scan(pdf=first_signal_pdf,
                                                      data=data,
                                                      mu_bounds=mu_bounds_excl,
                                                      mu_step=mu_step_excl)

    fig, axes = hep.plot.brazil(x=test_mus, exp=cls_exp, obs=cls_obs,
                     xlabel=r'$\mu$',
                     ylabel=r'$\mathrm{CLs}$',
                     xlim=mu_bounds_excl,
                     ylim=(0.0, 1.0),
                     yline=test_size)
    assert fig

    mu_excl_obs, mu_excl_exp = hep.stats.invert_interval(cls_obs, cls_exp, test_mus, test_size=test_size)
    assert np.isclose(mu_excl_obs,  0.55788, rtol=0, atol=0.001)

