"""
Preprocess data to get the npz file.
"""
import numpy as np
import phate
import scprep
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist

def get_data_dict(data, colors):
    phate_operator = phate.PHATE(knn=4, decay=15, t=12, n_jobs=-2)
    Y_phate = phate_operator.fit_transform(data)
    X_dist = squareform(pdist(phate_operator.diff_potential))
    return dict(data=data, phate=Y_phate, dist=X_dist, colors=colors)