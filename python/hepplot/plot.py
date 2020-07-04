"""
hepplot/plot.py

See:
https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py

TODOs:
-   Support support arrays as input and convert internal
    calculations to use numpy.
-   Make sure everything can be optional: y, yerr, yerrs, data.
-   Support symmetric gaussian errors for yerr and yerrs.
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

from . import stat


#plt.style.use(hep.style.ATLAS)
plt.style.use([hep.style.CMS, hep.style.firamath])


def hist1d(bins,
           y=None,
           yerr=None,
           yerrs=None,
           data=None,
           labels=None,
           ratio=False,
           unit='GeV'):

    n_bins = len(bins)-1
    n_samples = 0

    if y is not None:
        n_samples = len(y)
        y = y.copy()
        y.reverse()
        if labels is not None:
            labels = labels.copy()
            labels.reverse()

        for y_i in y:
            assert n_bins == len(y_i)

#    TODO: convert to and support numpy arrays as input
#    bins = np.asarray(bins, dtype=np.float32)
#    if y is not None:
#        y = np.asarray(y, dtype=np.float32)
#    if yerr is not None:
#        yerr = np.asarray(yerr, dtype=np.float32)
#    if yerrs is not None:
#        yerrs = np.asarray(yerrs, dtype=np.float32)
#    if data is not None:
#        data = np.asarray(data, dtype=np.float32)

    ## convert yerr from [bin][up,down] to [down,up][bin]
    if yerr is not None:
        yerr = [ 
                [ _y[1] for _y in yerr ],
                [ _y[0] for _y in yerr ],
               ]
#        yerr = np.swapaxes(yerr, 0, 1)
#        yerr = np.flip(yerr, 0)

    ## convert yerrs from [sample][bin][up,down] to [sample][down,up][bin]
    if yerrs is not None:
        yerrs = [ 
                    [
                        [ _y[1] for _y in _yerr ],
                        [ _y[0] for _y in _yerr ],
                    ] for _yerr in yerrs
               ]
#        yerrs = np.swapaxes(yerrs, 1, 2)
#        yerrs = np.flip(yerr, 1)

    ## prep derivative data
    ytotal = None
    if y is not None:
        ytotal  = [sum(i) for i in zip(*y)]
#        ytotal = np.sum(y, axis=1)
    if data is not None:
        dataerr = [
                    [stat.poisson_error_down(_y) for _y in data],
                    [stat.poisson_error_up(_y) for _y in data],
                  ]
#        dataerr = np.zeros((2, n_bins), dtype=np.float32)
#        for j_bin in range(n_bins):
#            dataerr[0][j_bin] = stat.poisson_error_down(data[j_bin])
#            dataerr[1][j_bin] = stat.poisson_error_up(data[j_bin])

    ## make top subplot
    fig = plt.figure()
    axes = list()
    if (data is not None) and ratio:
        gs = fig.add_gridspec(2, 1, height_ratios=(3, 1),
                              wspace=0, hspace=0.04)
        ax1 = fig.add_subplot(gs[0, 0])
        axes.append(ax1)
    else:
        fig, ax1 = plt.subplots()
        axes.append(ax1)
    
    ## plot hist stack
    if y is not None:
        colors = [plt.cm.Spectral(i/float(n_samples-1)) for i in range(n_samples)]
        binned = [bins[:-1] for _ in range(n_samples)]
#        binned = np.asarray([bins[:-1] for _ in range(n_samples)], dtype=np.float32)
        weights = y
        bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)
        binwidths = [bins[i+1]-bins[i] for i in range(n_bins)]
#        binwidths = np.asarray([bins[i+1]-bins[i] for i in range(n_bins)], dtype=np.float32)

#        print('DEBUG: n_samples', n_samples, flush=True)
#        print('DEBUG: bins.shape', bins.shape, flush=True)
#        print('DEBUG: binned.shape', binned.shape, flush=True)
#        print('DEBUG: weights.shape', weights.shape, flush=True)
        
        plt.hist(binned, bins,
            weights=weights,
            stacked=True,
            density=False,
            color=colors,
            label=labels,
            )
    
    ## sum yerrs to yerr
    if yerrs is not None:
        for yerrs_i in yerrs:
            assert len(yerrs_i) == 2
        yerr_down = [0.]*n_bins
        for yerrs_i in yerrs:
            assert len(yerrs_i[0]) == n_bins
            for j_yerr, yerr_j in enumerate(yerrs_i[0]):
                yerr_down[j_yerr] += yerr_j*yerr_j
        for j_yerr in range(n_bins):
            yerr_down[j_yerr] = math.sqrt(yerr_down[j_yerr])
        yerr_up = [0.]*n_bins
        for yerrs_i in yerrs:
            assert len(yerrs_i[1]) == n_bins
            for j_yerr, yerr_j in enumerate(yerrs_i[1]):
                yerr_up[j_yerr] += yerr_j*yerr_j
        for j_yerr in range(n_bins):
            yerr_up[j_yerr] = math.sqrt(yerr_up[j_yerr])
        yerr = [yerr_down, yerr_up]
#        yerr = np.sqrt(np.sum(yerrs**2, axis=0))

    ## plot error band
    if yerr is not None:
        xerr = [
                [w/2 for w in binwidths],
                [w/2 for w in binwidths],
            ]
        xerr = np.asarray(xerr)
        uncert_boxes = make_error_boxes(ax1, bincenters, ytotal, xerr, yerr,
                                        hatch='///')
    
    ## plot data
    if data is not None:
        plt.errorbar(bincenters, data, yerr=dataerr, 
            label='Data',
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=2,
            capsize=0,
            markersize=8,
            zorder=100,
            )
    
    ## axis labels
    plt.ylabel('Events / (%g %s)' % (binwidths[0], unit))
    if not ratio:
        plt.xlabel('Dependent variable [%s]' % (unit))

    ## make legend
    total_mean = sum([y_i*x_i for y_i, x_i in zip(ytotal, bincenters)])/n_bins
    middle_of_range = (bins[-1] - bins[0])/2
    
    leg_handles, leg_labels = ax1.get_legend_handles_labels()

    if data is not None:
        data_handle = leg_handles.pop()
        data_label = leg_labels.pop()
        assert data_label == 'Data'

    leg_handles.reverse()
    leg_labels.reverse()

    if yerr is not None:
        leg_handles.append(Patch(facecolor='darkgray',
                                 alpha=0.4,
                                 hatch='///'))
        leg_labels.append('Uncert.')

    if data is not None:
        leg_handles.append(data_handle)
        leg_labels.append(data_label)
    
    leg_loc = 'upper left'
    if total_mean > middle_of_range:
        leg_loc = 'upper right'

    leg = plt.legend(leg_handles, leg_labels, loc=leg_loc)
    
    ## make ratio
    if (data is not None) and ratio:
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        axes.append(ax2)
        
        plt.axhline(y=1.0, color='darkgray', linestyle='-', zorder=-1)
        
        y_ratio = [d_i/y_i if y_i else 0. for d_i, y_i in zip(data, ytotal)]
        y_ratio_err = [
            [de_i/d_i for d_i, de_i in zip(data, dataerr[0])],
            [de_i/d_i for d_i, de_i in zip(data, dataerr[1])],
        ]

        ## plot error band
        if yerr is not None:
            y_ratio_band = [
                [ye_i/y_i for y_i, ye_i in zip(ytotal, yerr[0])],
                [ye_i/y_i for y_i, ye_i in zip(ytotal, yerr[1])],
            ]
        
            xerr = [
                    [w/2 for w in binwidths],
                    [w/2 for w in binwidths],
                ]
            make_error_boxes(ax2, bincenters, [1.0]*n_bins, xerr, y_ratio_band,
                             hatch='///')
        
        ## plot ratio
        plt.errorbar(bincenters, y_ratio, yerr=y_ratio_err,
            label='ratio',
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=2,
            capsize=0,
            markersize=8,
            zorder=100,
            )
        
        ## axis labels
        ax2.set_ylabel('Data / Model')
        #ax2.set_ylim(0.7, 1.3) # HACK
        plt.xlabel('Dependent variable [unit]')
    
        fig.subplots_adjust(wspace=0, hspace=0)


def make_error_boxes(ax, xdata, ydata, xerror, yerror,
                     facecolor='darkgray',
#                     edgecolor='none',
                     alpha=0.4,
                     hatch=None,
                     zorder=20):
    """
    From:
    https://matplotlib.org/3.1.0/gallery/statistics/errorbars_and_boxes.html
    """

    # Create list for all the error patches
    errorboxes = []

    xerror = np.asarray(xerror)
    yerror = np.asarray(yerror)

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes,
                         facecolor=facecolor,
                         alpha=alpha,
#                         edgecolor=edgecolor,
                         hatch=hatch,
                         zorder=zorder)

    # Add collection to axes
    ax.add_collection(pc)

    return pc
