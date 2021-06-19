import matplotlib.pyplot as plt
import numpy as np 

from sklearn.metrics import roc_curve
def ROC(y_true, y_reco):
    fig, ax = plt.subplots(figsize = (9,6))

    x, y, _ = roc_curve(y_true, y_reco)

    ax.plot(x, y, "k")
    ax.set_title("AUC")
    ax.set_xlabel("TPR")
    ax.set_ylabel("FPR")

    return fig

def zenith_histogram_from_angle(y_true, y_reco):
    zenith_reco = y_reco[:, 0]
    zenith_true = y_true[:, 1]

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.set_title("Zenith distributions")
    ax.set_xlabel("Zenith")
    ax.set_ylabel("Freq")

    ax.hist(zenith_reco, range = (0, np.pi), bins = 100, label = "reco", histtype= "step")
    ax.hist(zenith_true, range = (0, np.pi), bins = 100, label = "reco", histtype= "step")

    ax.legend()

    return fig

def zenith_histogram_from_z(y_true, y_reco):
    zenith_reco = np.arccos(y_reco[:, 2])
    zenith_true = np.arccos(y_true[:, 2])

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.set_title("Zenith distributions")
    ax.set_xlabel("Zenith")
    ax.set_ylabel("Freq")

    ax.hist(zenith_reco, range = (0, np.pi), bins = 100, label = "reco", histtype= "step")
    ax.hist(zenith_true, range = (0, np.pi), bins = 100, label = "true", histtype= "step")

    ax.legend()

    return fig

def zenith_2d_hist_from_angle(y_true, y_reco):
    zenith_reco = y_reco[:, 0]
    zenith_true = y_true[:, 1]

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.set_title("Zenith true-reco")
    ax.set_xlabel("True")
    ax.set_ylabel("Reco")

    ax.hist2d(zenith_true, zenith_reco, bins = 100, range = ((0, np.pi), (0, np.pi)))

    return fig

def zenith_2d_hist_from_z(y_true, y_reco):
    zenith_reco = np.arccos(y_reco[:, 2])
    zenith_true = np.arccos(y_true[:, 2])

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.set_title("Zenith true-reco")
    ax.set_xlabel("True")
    ax.set_ylabel("Reco")

    ax.hist2d(zenith_true, zenith_reco, bins = 100, range = ((0, np.pi), (0, np.pi)))

    return fig



