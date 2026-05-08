# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 17:19:58 2025

@author: malte
"""
from pathlib import Path
from evoaudio.pitch import Pitch
import numpy as np
from itertools import combinations
from typing import Callable, Set, List, Tuple, Dict
from enum import IntEnum
import pandas as pd
from itertools import permutations
from scipy.stats import wasserstein_distance
from evoaudio.individual import BaseIndividual
from evoaudio.population import Population

def extract_instruments(individual:BaseIndividual):
    seen_instruments = []
    for sample in individual.samples:
        if sample.instrument not in seen_instruments:
            seen_instruments.append(sample.instrument)
    return seen_instruments

def load_instrument_distances(path):
    file_path = Path(path)  # Normalize path for any OS
    distance_df = pd.read_csv(file_path, index_col=0)

    # Standardize instrument names in both index and columns
    replacements = {
        r'^Bouzouki$': 'BouzoukiSakis',
        r'^Tampura$': 'TampuraPAndBrothers',
        r'^Kantele$': 'KanteleLowWideRange',
        r'^Cumbus$': 'CumbusOpenStrings',
        r'^JinghuOperaViolin$': 'JinghuOperaviolin',
    }

    for pattern, replacement in replacements.items():
        distance_df.index = distance_df.index.str.replace(pattern, replacement, regex=True)
        distance_df.columns = distance_df.columns.str.replace(pattern, replacement, regex=True)

    return distance_df

def GDp_from_matrix(D: np.ndarray, p: float = 1) -> float:
    """
    Forward averaged Hausdorff distance from U to V using a precomputed distance matrix D.
    D[i][j] is the distance from u_i ∈ U to v_j ∈ V.
    """
    if D.size == 0 or D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError("Distance matrix must be non-empty")
    
    m = D.shape[0]  # Number of elements in U
    min_d_p = np.min(D ** p, axis=1)  # For each u ∈ U, find min dist to any v ∈ V
    return (np.sum(min_d_p) / m) ** (1 / p)

def avg_hausdorff_from_matrix(D: np.ndarray, p: float = 1) -> float:
    """
    Symmetric averaged Hausdorff distance using precomputed distances.
    D[i][j] = distance between u_i ∈ U and v_j ∈ V
    """
    return max(GDp_from_matrix(D, p), GDp_from_matrix(D.T, p))

def extract_submatrix_from_df(distance_df: pd.DataFrame,
                               predictions: list[str],
                               annotations: list[str]) -> pd.DataFrame:
    """
    Extract a submatrix of distances from a labeled DataFrame.

    distance_df: DataFrame where rows and columns are instrument names.
    predictions: list of instrument names (rows).
    annotations: list of instrument names (columns).

    Returns:
        Sub-DataFrame with shape (len(predictions), len(annotations))
    """

    # Ensure all labels exist
    missing_preds = [p for p in predictions if p not in distance_df.index]
    missing_annots = [a for a in annotations if a not in distance_df.columns]

    if missing_preds or missing_annots:
        raise ValueError(f"Missing labels — Predictions: {missing_preds}, Annotations: {missing_annots}")

    return distance_df.loc[predictions, annotations]

def avg_hausdorff_instruments(pop: Population,
                              annotation:list[tuple],
                              p: float = 1) -> float:
    """
    Symmetric averaged Hausdorff distance using precomputed distances.
    D[i][j] = distance between u_i ∈ U and v_j ∈ V
    """
    return avg_hausdorff_instruments_full_piece(pop, {0: annotation})


def avg_hausdorff_instruments_full_piece(pop: Population,
                              annotations:dict[int,list[tuple]],
                              p: float = 1) -> float:
    """
    Symmetric averaged Hausdorff distance using precomputed distances.
    D[i][j] = distance between u_i ∈ U and v_j ∈ V
    """
    distance_df = load_instrument_distances(
        Path("./instrument_distances/mcadams_-1/average_absolute_distance/distance_matrix.csv")
    )
    error_per_onset = []
    time_onsets = list(annotations.keys())
    for i, onset in enumerate(pop.archive):
        individual = pop.archive[onset].individual
        time_onset = time_onsets[i]
        extracted_features = extract_instruments(individual)
        annotated_features = [annotation[0] for annotation in annotations[time_onset]]
        error_per_onset.append(avg_hausdorff_from_matrix(extract_submatrix_from_df(distance_df, extracted_features, annotated_features), p))
    return np.mean(error_per_onset)

def hausdorff_instruments_full_piece_GD_IGD(
        pop: Population,
        annotations: dict[float, list[tuple]],
        p: float = 1,
        algorithm: str = "MOEA"
    ):

    distance_df = load_instrument_distances(
        Path("./instrument_distances/mcadams_-1/average_absolute_distance/distance_matrix.csv")
    )

    time_onsets = list(annotations.keys())

    # ---- Build reference set V ----
    V = []
    for t in time_onsets:
        annotated_features = [a[0] for a in annotations[t]]
        V.append(annotated_features)

    # ---- Build population-induced set U ----
    U = []
    
    if algorithm=="EA":
        for i, onset in enumerate(pop.archive):
            extracted_features = extract_instruments(pop.archive[onset].individual)
            U.append(extracted_features)
    else:     
        for ind in pop.individuals:
            extracted_features = extract_instruments(ind)
            U.append(extracted_features)

    N = len(U)
    M = len(V)

    # ---- Build pairwise distance matrix ----
    D = np.zeros((N, M))

    for i, u in enumerate(U):
        print((i+1)/len(U))
        for j, v in enumerate(V):
            submatrix = extract_submatrix_from_df(
                distance_df,
                u,
                v
            )

            D[i, j] = avg_hausdorff_from_matrix(submatrix, p)

    # ---- Standard GD ----
    GD = (np.mean(np.min(D, axis=1)**p))**(1/p)

    # ---- Standard IGD ----
    IGD = (np.mean(np.min(D, axis=0)**p))**(1/p)

    return GD, IGD


