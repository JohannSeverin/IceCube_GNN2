import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1

def performance(diffs, Ns, ax = None):  
    if not ax:
        fig, ax1   = plt.subplots(figsize = (9, 6))
    else:
        ax1 = ax
    
    ax2        = ax1.twinx()
    
    
    splits = 15

    count, bins, _ = ax1.hist(np.log10(Ns), bins = splits, log = "true", color = "gray", zorder = 10)
    ax1.set_xlabel("$log_{10}N$")
    
    med_arr    = []
    perc_arr   = []

    for i in range(len(bins) - 1):
        current_diffs = diffs[(bins[i] < np.log10(Ns)) & ( np.log10(Ns) <= bins[i + 1])]
        median        = np.percentile(abs(current_diffs), 50)
        percentiles   = np.percentile(abs(current_diffs), [50 - 34, 50 + 34])

        med_arr.append(median)
        perc_arr.append(percentiles)

    xs = (bins[1:] + bins[:-1]) / 2

    ax2.plot(xs, med_arr, "ro", zorder = -1)
    ax2.errorbar(xs, med_arr, np.array(perc_arr).T, linestyle = "None", color = "red", capsize = 2)
    
    return ax1, ax2
    

def reco_true_hist(reco, true, ax = None):
    if not ax:
        fig, ax = plt.subplots(figsize = (7, 7))
    
    h, x_edge, y_edge, _ = ax.hist2d(true, reco, bins = 50)
    
    ax.set_xlabel("True")
#     ax.set_ylabel("Reconstructed")
    
    return ax


def azi_zen_from_units(arr):
    azi = np.arctan2(arr[:, 1], arr[:, 0])
    zen = np.arccos(arr[:, 2])
    return azi, zen


def sig_from_k(k):
    return np.sqrt(-2 * np.log(i1(k)/i0(k)))

def angle_diffs(A, B):
    return np.arcsin(np.sin(A) * np.cos(B) - np.cos(A) * np.sin(B))