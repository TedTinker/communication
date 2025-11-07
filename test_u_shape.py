#%% 

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal
from matplotlib.ticker import FuncFormatter
from itertools import accumulate
from math import log
from scipy import interpolate
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression
from itertools import accumulate
from statistics import mode
from collections import Counter

from utils import load_dicts, rolling_average

# This file tests the impact of exceptions.



rolling_window = 10000

# Assuming the arguments contains two arg_names: one for exceptions, one for pseudo-exceptions, in that order.
plot_dicts, min_max_dict, complete_order = load_dicts({"titles" : ["ef_q2t_5", "ef_q2t_6"]})

print("Getting results with exceptions...")
exceptions_dict = plot_dicts[0]
exceptions = [rolling_average(wins, window_size=rolling_window) for wins in exceptions_dict["wins_exception"]]  

print("\nGetting results with pseudo-exceptions...")
pseudo_exceptions_dict = plot_dicts[1]
pseudo_exceptions = [rolling_average(wins, window_size=rolling_window) for wins in pseudo_exceptions_dict["wins_exception"]]  

print("\nGot results!")

# So, we have two lists. Both lists have a list for each agent, showing rolling-average win-rates with exceptions.

#%%



def plot_these(list_1, list_2):
    fig, ax = plt.subplots(len(list_1), 2, figsize=(20, 4 * len(list_1)))
    fig.suptitle("Exceptions vs Pseudo-Exceptions", fontsize=16)

    # Ensure ax is a 2D array even if len(exceptions) == 1
    if len(exceptions) == 1:
        ax = np.expand_dims(ax, axis=0)

    # Define the plot function
    def plot_rolling_average_wins(axis, data, label, color):
        axis.plot([100 * d for d in data], label=label, color=color)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Success Rate")

    # Plot all agent data
    for i in range(len(list_1)):
        plot_rolling_average_wins(ax[i, 0], list_1[i], label="Exceptions", color="blue")
        plot_rolling_average_wins(ax[i, 1], list_2[i], label="Pseudo-Exceptions", color="green")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

plot_these(exceptions, pseudo_exceptions)


#%%



"""def smooth_all(data_list, frac=0.03, polyorder=1):
    smoothed = []
    for series in data_list:
        y = np.array(series, dtype=float)
        n = len(y)
        window = max(5, int(frac * n))
        if window % 2 == 0:
            window += 1
        if window >= n:
            window = n - 1 if n % 2 == 0 else n
        y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
        smoothed.append(y_smooth)
    return smoothed

smoothed_exceptions = smooth_all(exceptions)
smoothed_pseudo_exceptions = smooth_all(pseudo_exceptions)


plot_these(smoothed_exceptions, smoothed_pseudo_exceptions)"""



#%% 



def u_shape_score(
    y,
    burnin_frac=0.08,          # ignore the first ~8% of steps
    smooth_frac=0.03,          # light smoothing window (fraction of length)
    min_pos=(0.20, 0.80)       # interior window where we allow the minimum
):
    """
    Return a U-shapedness score in [0, 1].
    High = clear interior valley bracketed by high regions on both sides.
    Low = monotone-ish or minimum at an edge.

    Parameters are conservative defaults for long training curves (e.g., 60k points).
    """

    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 40:
        return 0.0

    # 1) Optional burn-in removal (prevents the very start from being the min)
    b = int(n * burnin_frac)
    if b < n - 10 and b > 0:
        y = y[b:]
        n = y.size

    # 2) Interior minimum
    i_min = int(np.argmin(y))
    lo = int(n * min_pos[0])
    hi = int(n * min_pos[1])
    if not (lo <= i_min <= hi):
        return 0.0

    # 3) Side peaks
    left = y[:i_min]
    right = y[i_min+1:]
    if left.size < 3 or right.size < 3:
        return 0.0

    iL = int(np.argmax(left))             # index in left segment
    iR = int(np.argmax(right))            # index in right segment
    yL = float(left[iL])
    yR = float(right[iR])
    yV = float(y[i_min])                  # valley

    # Depths above valley on both sides (normalized)
    dL = max(0.0, yL - yV)
    dR = max(0.0, yR - yV)
    depth = min(dL, dR)                   # both sides must be high
    if depth <= 1e-6:
        return 0.0

    # 4) Slope consistency (down then up) around the valley
    segL = y[iL:i_min+1]                  # from left peak to valley
    segR = y[i_min:i_min+1+iR+1]          # from valley to right peak
    decL = np.mean(np.diff(segL) < 0) if segL.size > 1 else 0.0
    incR = np.mean(np.diff(segR) > 0) if segR.size > 1 else 0.0
    slope_score = min(decL, incR)         # both runs should be mostly monotone in the expected direction

    # 5) Balance (penalize very lopsided dips)
    balance = min(dL, dR) / (max(dL, dR) + 1e-12)

    # 6) Width (avoid needle dips)
    # width = how many points from left peak to valley and valley to right peak, normalized
    width = min(i_min - iL, iR + 1) / n
    width = max(0.0, min(width / 0.20, 1.0))  # saturate when the valley spans ~20% of the series

    # 7) Combine — convex combination inside [0,1]
    score = float(depth * (0.5 * slope_score + 0.3 * balance + 0.2 * width))
    return score



exception_u_scores = [u_shape_score(e) for e in exceptions]
pseudo_exception_u_scores = [u_shape_score(e) for e in pseudo_exceptions]

print(exception_u_scores)
print(pseudo_exception_u_scores)



def compare_u_scores_ttest(group_a, group_b):
    t_stat, p_val_two_tailed = ttest_ind(group_a, group_b, equal_var=False)
    if t_stat > 0:
        p_val_one_tailed = p_val_two_tailed / 2
    else:
        p_val_one_tailed = 1 - p_val_two_tailed / 2  # not significant in this direction

    return {
        'p_value': p_val_one_tailed,
        't_statistic': t_stat,
        'mean_a': np.mean(group_a),
        'mean_b': np.mean(group_b),
        'effect_size': np.mean(group_a) - np.mean(group_b)
    }



results = compare_u_scores_ttest(exception_u_scores, pseudo_exception_u_scores)

print("Permutation Test Results:")
for key, val in results.items():
    print(f"{key}: {val:.4f}")
    
    
    
    


# %%





def analyze_u_shape(y, burnin_frac=0.08, min_pos=(0.20, 0.80)):
    """Return score and indices of left peak, valley, right peak (if valid)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 40:
        return 0.0, None, None, None

    b = int(n * burnin_frac)
    if b < n - 10 and b > 0:
        y = y[b:]
        n = y.size

    i_min = int(np.argmin(y))
    lo = int(n * min_pos[0])
    hi = int(n * min_pos[1])
    if not (lo <= i_min <= hi):
        return 0.0, None, None, None

    left = y[:i_min]
    right = y[i_min+1:]
    if left.size < 3 or right.size < 3:
        return 0.0, None, None, None

    iL = int(np.argmax(left))
    iR = int(np.argmax(right))
    yL, yR, yV = float(left[iL]), float(right[iR]), float(y[i_min])

    dL, dR = max(0, yL - yV), max(0, yR - yV)
    depth = min(dL, dR)
    if depth <= 1e-6:
        return 0.0, None, None, None

    segL, segR = y[iL:i_min+1], y[i_min:i_min+1+iR+1]
    decL = np.mean(np.diff(segL) < 0) if segL.size > 1 else 0.0
    incR = np.mean(np.diff(segR) > 0) if segR.size > 1 else 0.0
    slope_score = min(decL, incR)
    balance = min(dL, dR) / (max(dL, dR) + 1e-12)
    width = min(i_min - iL, iR + 1) / n
    width = max(0, min(width / 0.20, 1.0))
    score = float(depth * (0.5 * slope_score + 0.3 * balance + 0.2 * width))

    return score, iL + b, i_min + b, i_min + 1 + iR + b



def plot_these_with_u(list_1, list_2, label1="Exceptions", label2="Pseudo-Exceptions"):
    fig, ax = plt.subplots(len(list_1), 2, figsize=(20, 4 * len(list_1)))
    fig.suptitle("Exceptions vs Pseudo-Exceptions (U-shape annotated)", fontsize=16)

    if len(list_1) == 1:
        ax = np.expand_dims(ax, axis=0)

    def plot_agent(axis, y, color, label):
        y = np.asarray(y) * 100
        axis.plot(y, color=color, lw=1.5)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Success Rate")
        axis.set_title(label, fontsize=12)

        # Compute u-score and key points
        score, iL, iM, iR = analyze_u_shape(y)
        axis.text(0.01, 0.95, f"U-score: {score:.3f}",
                  transform=axis.transAxes, fontsize=10,
                  verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))

        # Annotate if valid
        if all(v is not None for v in [iL, iM, iR]):
            axis.axvline(iL, color="orange", ls="--", lw=1)
            axis.axvline(iM, color="red", ls="--", lw=1)
            axis.axvline(iR, color="orange", ls="--", lw=1)
            axis.plot([iL, iM, iR],
                      [y[iL], y[iM], y[iR]],
                      "ro", markersize=5)

        return score

    scores1, scores2 = [], []
    for i in range(len(list_1)):
        scores1.append(plot_agent(ax[i, 0], list_1[i], "blue", label1))
        scores2.append(plot_agent(ax[i, 1], list_2[i], "green", label2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    return scores1, scores2



exception_u_scores, pseudo_exception_u_scores = plot_these_with_u(exceptions, pseudo_exceptions)

results = compare_u_scores_ttest(exception_u_scores, pseudo_exception_u_scores)
print("Permutation Test Results:")
for key, val in results.items():
    print(f"{key}: {val:.4f}")