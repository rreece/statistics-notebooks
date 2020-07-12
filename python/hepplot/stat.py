"""
hepplot/stat.py

See:
https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
https://en.wikipedia.org/wiki/Chi-square_distribution#Asymptotic_properties
https://www.johndcook.com/blog/wilson_hilferty/
"""


import math
import numpy as np
import pyhf


def poisson_error_up(data):
    y1 = data + 1.0
    d = 1.0 - 1.0/(9.0*y1) + 1.0/(3*math.sqrt(y1))
    return y1*d*d*d-data


def poisson_error_down(data):
    y = data
    if y == 0.0: return 0.0
    d = 1.0 - 1.0/(9.0*y) - 1.0/(3.0*math.sqrt(y))
    return data-y*d*d*d


def make_pdf(bkg_data, bkg_uncerts, signal_data):
    pdf = pyhf.simplemodels.hepdata_like(signal_data=signal_data, bkg_data=bkg_data, bkg_uncerts=bkg_uncerts)
    return pdf


def hypo_test(pdf, data, poi=1.0, init_pars=None, par_bounds=None):
    if init_pars is None:
        init_pars = pdf.config.suggested_init()
    if par_bounds is None:
        par_bounds = pdf.config.suggested_bounds()
    CLs_obs, CLs_exp_band = pyhf.infer.hypotest(poi,
                                                data + pdf.config.auxdata,
                                                pdf,
                                                init_pars,
                                                par_bounds,
                                                return_expected_set=True)
    return CLs_obs, CLs_exp_band


def hypo_test_scan(pdf, data, test_pois=None, init_pars=None, par_bounds=None):
    if par_bounds is None:
        par_bounds = pdf.config.suggested_bounds()
    if test_pois is None:
        poi_bounds = par_bounds[pdf.config.poi_index]
        steps_per_mu = 10
        poi_steps = (poi_bounds[1] - poi_bounds[0]) * steps_per_mu + 1
        test_pois = np.linspace(poi_bounds[0], poi_bounds[1], poi_steps)
    hypo_tests = [
        hypo_test(pdf=pdf,
                  data=data,
                  poi=poi,
                  init_pars=init_pars,
                  par_bounds=par_bounds)
        for poi in test_pois
    ]
    cls_exp = [np.array([test[1][i] for test in hypo_tests]).flatten() for i in range(5)]
    cls_obs = np.array([test[0] for test in hypo_tests]).flatten()
    return hypo_tests, test_pois, cls_exp, cls_obs


def invert_interval(test_mus, cls_exp, cls_obs, test_size=0.05):
    crossing_test_stats = {'exp': [], 'obs': None}
    for cls_exp_sigma in cls_exp:
        crossing_test_stats['exp'].append(
            np.interp(
                test_size, list(reversed(cls_exp_sigma)), list(reversed(test_mus))
            )
        )
    crossing_test_stats['obs'] = np.interp(
        test_size, list(reversed(cls_obs)), list(reversed(test_mus))
    )
    return crossing_test_stats
