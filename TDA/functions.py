## TDA-based feature extraction and classification for Feature Maps. 
## By Matas Vaitkevicius, Spring 2026
## TODO:


""" (no order of importance)
1. Tune the thresholds in ClassFromFeats() for better blob/edge classification.
2. Explore additional TDA features (e.g., persistence landscapes, Betti curves).
3. Explore a variable alpha for tau_frac based on the feature map's characteristics (e.g., variance).
4. Add feature to compare topology of maps and rate some similarity score (e.g., Wasserstein distance between persistence diagrams).
"""

"""
Tuning class from feats, collect data from a whole set of data then set thresholds for blob/edge classification.

For alpha determine some distrubution of persistence values in and exclude the lowest quartile of values, so statstically only the top 75%
is kept for classification.
"""


import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pickle
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import gudhi as gd
from pathlib import Path
from scipy.special import entr


##Computation of TDA and Classes, Visualization accesible with plot_scatter and animate.
#

def ComputeFeats(feature_map, tau_frac=0.03, superlevel=True, plot_scatter=False, animate=False):


    """
    Computer perstience-based features from a 2D scalar field (feature map).

    Parameters
    ----------
    feature_map : (H,W) array
        Scalar field (feature map).
    tau_frac : float
        Persistence threshold = tau_frac * (range of the resized feature map).
        Increasing tau_frac discards more small/noisy features.
    superlevel : bool
        If True, analyze superlevel sets {feature >= t} by computing sublevel PH of (-feature).
    Returns
    -------
    feats : np.ndarray shape (6,)
        [maxp0, n0_tau, sump0_tau, maxp1, n1_tau, sump1_tau]
    tau : float
        The numeric persistence cutoff used.
    """
    

    fm = np.asarray(feature_map)

    rnge = float(fm.max() - fm.min())
    tau = tau_frac * rnge if rnge > 0 else 0.0

    f = fm
    cc = gd.CubicalComplex(top_dimensional_cells=f)
    cc.persistence()

    D0 = np.asarray(cc.persistence_intervals_in_dimension(0), dtype=np.float64)
    D1 = np.asarray(cc.persistence_intervals_in_dimension(1), dtype=np.float64)


    def stats(D):
        if D.size == 0:
            return 0.0, 0, 0.0
        pers = D[:, 1] - D[:, 0]
        pers = pers[np.isfinite(pers)]
        if pers.size == 0:
            return 0.0, 0, 0.0
        mask = pers >= tau
        return float(pers.max()), int(mask.sum()), float(pers[mask].sum()) if mask.any() else 0.0

    maxp0, n0, sump0 = stats(D0)
    maxp1, n1, sump1 = stats(D1)

    if plot_scatter:
        def ScatterPlotBirthDeath(D0, D1, tau, title="Birth-Death Scatter Plot"):
            plt.figure(figsize=(6, 6))
            if D0.size > 0:
                plt.scatter(D0[:, 0], D0[:, 1], color='blue', label='H0', alpha=0.5)
            if D1.size > 0:
                plt.scatter(D1[:, 0], D1[:, 1], color='red', label='H1', alpha=0.5)
            plt.plot([D0[:, 0].min(), D0[:, 1].max()], [D0[:, 0].min(), D0[:, 1].max()], 'k--')
            plt.axhline(y=tau, color='gray', linestyle='--')
            plt.axvline(x=tau, color='gray', linestyle='--')
            plt.xlabel('Birth')
            plt.ylabel('Death')
            plt.title(title)
            plt.legend()
            plt.grid()
            plt.show()

    if animate:
        def AnimateFiltration(feature_map, D0, D1, tau, title="Filtration Animation"):
            thresholds = np.linspace(feature_map.max(), feature_map.min(), 100)
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow((feature_map >= thresholds[0]).astype(float), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{title}\nt = {thresholds[0]:.3f}")
            ax.axis("off")

            def update(frame_idx):
                t = thresholds[frame_idx]
                binary = (feature_map >= t).astype(float)
                im.set_data(binary)
                ax.set_title(f"{title}\nt = {t:.3f}")
                return [im]

            anim = FuncAnimation(fig, update, frames=len(thresholds), interval=100, blit=False)
            plt.show()

    feats = np.array([maxp0, n0, sump0, maxp1, n1, sump1], dtype=np.float64)
    return feats, tau, D0, D1

#classify the feature map into blob, edge, or mixed based on the computed TDA features.

def ClassFromFeats(feats,
                blob_score_min=5.0,
                blob_n0_max=12,
                blob_n1_max=6,
                edge_n0_min=60,
                edge_n1_min=25,
                edge_score_min=1):
    """
    Classify a layer's topology into {'blob','edge','mixed'}.

    feats = [maxp0, n0_tau, sump0_tau, maxp1, n1_tau, sump1_tau]

    Interpretation:
      - H0 counts (n0_tau) large => many persistent connected components => fragmented/edge-like.
      - H1 counts (n1_tau) large => many persistent holes/loops => edge-like textures.
      - maxp0 large relative to n0_tau => one dominant region => blob-like.

    Returns
    -------
    label : str
        'blob' | 'edge' | 'mixed'
    diag : dict
        scores and the raw numbers for debugging / threshold tuning.
    """
    maxp0, n0, sump0, maxp1, n1, sump1 = map(float, feats)

    blob_score = maxp0 / (n0 + 1.0)
    edge_score = n1 / (n0 + 1.0)

    # Blob: few components, few loops, dominant H0 feature
    if (blob_score >= blob_score_min) and (n0 <= blob_n0_max) and (n1 <= blob_n1_max):
        label = "blob"
    # Edge: many components or many loops, or strong edge_score
    elif (n1 >= edge_n1_min) and (edge_score >= edge_score_min):
        label = "edge"
    else:
        label = "mixed"

    diag = dict(
        maxp0=maxp0, n0=n0, sump0=sump0,
        maxp1=maxp1, n1=n1, sump1=sump1,
        blob_score=blob_score, edge_score=edge_score
    )
    return label, diag




