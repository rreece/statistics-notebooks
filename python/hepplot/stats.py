"""
hepplot/stats.py
"""

import math
import numpy as np
import pyhf
import scipy.stats


#------------------------------------------------
# Confidence interval helper functions
#------------------------------------------------

def poisson_error_up(data):
    """
    See:
    https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
    https://en.wikipedia.org/wiki/Chi-square_distribution#Asymptotic_properties
    https://www.johndcook.com/blog/wilson_hilferty/
    """
    y1 = data + 1.0
    d = 1.0 - 1.0/(9.0*y1) + 1.0/(3*math.sqrt(y1))
    return y1*d*d*d-data


def poisson_error_down(data):
    y = data
    if y == 0.0: return 0.0
    d = 1.0 - 1.0/(9.0*y) - 1.0/(3.0*math.sqrt(y))
    return data-y*d*d*d


def clopper_pearson(k, n, cl):
    """
    https://gist.github.com/DavidWalz/8538435
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    alpha = 1 - cl
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


#------------------------------------------------
# pyhf helpers
#------------------------------------------------

def make_model(bkg_data, bkg_uncerts, signal_data):
    model = pyhf.simplemodels.hepdata_like(signal_data=signal_data,
                                           bkg_data=bkg_data,
                                           bkg_uncerts=bkg_uncerts)
    return model


def hypo_test(model, data, mu=1.0, init_pars=None, par_bounds=None):
    if init_pars is None:
        init_pars = model.config.suggested_init()
    if par_bounds is None:
        par_bounds = model.config.suggested_bounds()
    CLs_obs, CLs_exp_band = pyhf.infer.hypotest(mu,
                                                data + model.config.auxdata,
                                                model,
                                                init_pars,
                                                par_bounds,
                                                return_expected_set=True)
    return CLs_obs, CLs_exp_band


def hypo_test_scan(model, 
                   data,
                   test_mus=None,
                   init_pars=None,
                   mu_bounds=None,
                   mu_step=0.1,
                   ):
    par_bounds = model.config.suggested_bounds()
    if mu_bounds is None:
        mu_bounds = par_bounds[model.config.poi_index]
    else:
        par_bounds[model.config.poi_index] = mu_bounds
    if test_mus is None:
        mu_steps = (mu_bounds[1] - mu_bounds[0]) * int(round(1./mu_step)) + 1
        test_mus = np.linspace(mu_bounds[0], mu_bounds[1], mu_steps)
    hypo_tests = [
        hypo_test(model=model,
                  data=data,
                  mu=mu,
                  init_pars=init_pars,
                  par_bounds=par_bounds)
        for mu in test_mus
    ]
    return hypo_tests, test_mus


def invert_interval(hypo_tests, test_mus, test_size=0.05):
    cls_exp = [np.array([test[1][i] for test in hypo_tests]).flatten() for i in range(5)]
    cls_obs = np.array([test[0] for test in hypo_tests]).flatten()
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
