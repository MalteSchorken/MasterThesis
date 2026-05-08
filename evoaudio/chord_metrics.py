# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 17:26:36 2025

@author: malte
"""

from evoaudio.pitch import Pitch
import numpy as np
from itertools import combinations
from typing import Callable, Set, List, Tuple, Dict
from enum import IntEnum
from itertools import permutations
from scipy.stats import wasserstein_distance
from evoaudio.individual import BaseIndividual
from evoaudio.population import Population


def extract_pitches(individual:BaseIndividual):
    seen_pitches = []
    for sample in individual.samples:
        if sample.pitch not in seen_pitches:
            seen_pitches.append(sample.pitch)
    return seen_pitches

def pcs(pitches: Set[Pitch]) -> Set[int]:
    return {p % 12 for p in pitches}

def full_pitch_vector(chord: Set[IntEnum]) -> np.ndarray:
    """Returns a 12D binary vector for pitch class content."""
    pc_vec = np.zeros((109,), dtype=int)
    for pitch in chord:
        pc_vec[pitch] = 1
    return pc_vec

def pitch_class_vector(chord: Set[IntEnum]) -> np.ndarray:
    """Returns a 12D binary vector for pitch class content."""
    pc_vec = np.zeros((12,), dtype=int)
    for pitch in chord:
        pc_vec[pitch % 12] = 1
    return pc_vec

def cyclic_pc_dist(u: int, v: int) -> int:
    return min((u - v) % 12, (v - u) % 12)

def build_fifth_index_map():
    """
    Build a dict: pitch class (0-11) → index on the spiral of fifths (relative to C).
    """
    fifth_map = {}
    for i in range(12):
        pc = (7 * i) % 12
        #if i > 6:
        #    i = i-12
        if pc not in fifth_map:
            fifth_map[pc] = i
    return fifth_map

PITCH_CLASS_TO_FIFTH = build_fifth_index_map()

def pitch_class_to_fifth_index(pc):
    """
    Convert pitch class to its index along the fifths spiral.
    """
    pc_mod = pc % 12
    return PITCH_CLASS_TO_FIFTH[pc_mod]

def pitch_to_point(k, r=1.0, h=0.5):
    """Convert fifth index k to 3D point on the spiral."""
    angle = k * np.pi / 2
    x = r * np.sin(angle)
    y = r * np.cos(angle)
    z = k * h
    return np.array([x, y, z])

def chord_to_centroid(chord_pcs, r=1.0, h=0.5):
    """
    Convert a chord (given as pitch classes or MIDI notes) to its spiral centroid.
    """
    fifth_indices = [pitch_class_to_fifth_index(p % 12) for p in chord_pcs]
    points = [pitch_to_point(k, r=r, h=h) for k in fifth_indices]
    centroid = np.mean(points, axis=0)
    return centroid

def spiral_distance(chord1, chord2, r=1.0, h=0.5):
    """
    Compute geometric spiral distance between two chords (given as pitch classes or MIDI).
    """
    c1 = chord_to_centroid(chord1, r=r, h=h)
    c2 = chord_to_centroid(chord2, r=r, h=h)
    return np.linalg.norm(c1 - c2)

def A_n_distance(A: Set[int], B: Set[int], p: float = 2.0) -> float:
    """Computes the minimal ℓ_p norm between pitch-class sets A and B (mod 12) with circular distance."""
    if len(A) != len(B):
        raise ValueError("Sets must have the same cardinality")

    A_list = list(A)
    B_list = list(B)

    min_total = float('inf')
    for perm in permutations(B_list):
        total = sum(cyclic_pc_dist(a, b) ** p for a, b in zip(A_list, perm))
        norm = total ** (1 / p)
        if norm < min_total:
            min_total = norm

    return min_total

def intervalic_distance(c1: Set[Pitch], c2: Set[Pitch]) -> int:
    def iv(pcset):
        vec = [0]*6
        for x,y in combinations(pcset,2):
            d = min((y-x) % 12, (x-y)%12)
            if 1 <= d <= 6: vec[d-1]+=1
        return vec
    iv1, iv2 = iv(pcs(c1)), iv(pcs(c2))
    return sum(abs(a-b) for a,b in zip(iv1, iv2))

def lp_distance_12(chord1: Set[Pitch], chord2: Set[Pitch], p: float = 2) -> float:
    """Generalized p-norm (Minkowski) distance between pitch-class sets."""
    v1 = pitch_class_vector(chord1)
    v2 = pitch_class_vector(chord2)
    return np.linalg.norm((v1-v2), ord=p)

def lp_distance_all(chord1: Set[Pitch], chord2: Set[Pitch], p: float = 2) -> float:
    """Generalized p-norm (Minkowski) distance between pitch-class sets."""
    v1 = full_pitch_vector(chord1)
    v2 = full_pitch_vector(chord2)
    return np.linalg.norm((v1-v2), ord=p)

def GDp(c1: Set[Pitch], c2: Set[Pitch], p: float = 2) -> float:
    """Forward averaged Hausdorff component from U to V."""
    U_pc = sorted(pcs(c1))
    V_pc = sorted(pcs(c2))
    m = len(U_pc)
    if m == 0 or not V_pc:
        print(f"  U_pc (extracted): {U_pc}")
        print(f"  V_pc (annotated): {V_pc}")
        print(f"  Original c1: {c1}")
        print(f"  Original c2: {c2}")
        raise ValueError("Both sets must be non-empty")
    total = 0.0
    for u in U_pc:
        min_d_p = min(cyclic_pc_dist(u, v)**p for v in V_pc)
        total += min_d_p
    return (total / m) ** (1 / p)

def avg_hausdorff_p(U: Set[Pitch], V: Set[Pitch], p: float = 2) -> float:
    """Symmetric (two-sided) averaged Hausdorff distance Δ_p(U, V)."""
    return max(GDp(U, V, p), GDp(V, U, p))

def projection_distance_general(
    U: Set[Pitch],
    V: Set[Pitch],
    distance_func: Callable[[Set[int], Set[int]], float] = A_n_distance,
    p: float = 1
) -> float:

    U_pc = {u % 12 for u in U}
    V_pc = {v % 12 for v in V}
    
    m, n = len(U_pc), len(V_pc)
    if n < m:
        temp = U_pc
        U_pc = V_pc
        V_pc = temp

    max_dist = 0.0
    for combo in combinations(list(V_pc), min(m,n)):
        S_pc = {v % 12 for v in combo}
        dist = distance_func(U_pc, S_pc, p)
        max_dist = max(max_dist, dist)

    return max_dist

def fujita_distance(U: Set[int], V: Set[int]) -> float:
    """
    Fujita distance (2024): asymmetric double average over pitch-class differences
    using circular mod-12 distances and symmetric penalty for difference.
    """
    U = {u % 12 for u in U}
    V = {v % 12 for v in V}
    
    if not U or not V:
        raise ValueError("Pitch-class sets must be non-empty.")

    union_size = len(U | V)
    if union_size == 0:
        return 0.0

    sum1 = sum(cyclic_pc_dist(u, v) for u in U for v in (V - U))
    sum2 = sum(cyclic_pc_dist(u, v) for u in (U - V) for v in V)

    part1 = sum1 / (union_size * len(U)) if len(U) > 0 else 0
    part2 = sum2 / (union_size * len(V)) if len(V) > 0 else 0

    return part1 + part2

def wasserstein(U: Set[IntEnum], V: Set[IntEnum]) -> float:
    """
    Wasserstein-1 distance (EMD) between two sets of Pitch enums (non-cyclic).
    Assumes uniform distribution over each chord.
    """
    X = [int(p) for p in U]
    Y = [int(p) for p in V]
    return wasserstein_distance(X, Y)

def Ginsel_dist(x,y):
    interval = abs(x - y)

    if interval == 0:
        return 0
    elif interval == 12:
        return 1
    elif interval == 7:
        return 2
    elif interval in {3, 4, 5, 9}:
        return 3
    elif interval in {1, 2, 6, 8, 10, 11}:
        return 4
    else:
        return 5

def GDp_G(c1: Set[Pitch], c2: Set[Pitch], p: float = 2) -> float:
    """Forward averaged Hausdorff component from U to V."""
    U_pc = [int(p) for p in c1]
    V_pc = [int(p) for p in c2]
    m = len(U_pc)
    if m == 0 or not V_pc:
        raise ValueError("Both sets must be non-empty")
    total = 0.0
    for u in U_pc:
        min_d_p = min(Ginsel_dist(u, v)**p for v in V_pc)
        total += min_d_p
    return (total / m) ** (1 / p)

def avg_hausdorff_p_G(U: Set[Pitch], V: Set[Pitch], p: float = 2) -> float:
    """Symmetric (two-sided) averaged Hausdorff distance Δ_p(U, V)."""
    return max(GDp_G(U, V, p), GDp_G(V, U, p))

def fujita_distance_G(U: Set[int], V: Set[int]) -> float:
    """
    Fujita distance (2024): asymmetric double average over pitch-class differences
    using circular mod-12 distances and symmetric penalty for difference.
    """
    U = {u for u in U}
    V = {v for v in V}
    
    if not U or not V:
        raise ValueError("Pitch-class sets must be non-empty.")

    union_size = len(U | V)
    if union_size == 0:
        return 0.0

    sum1 = sum(Ginsel_dist(u, v) for u in U for v in (V - U))
    sum2 = sum(Ginsel_dist(u, v) for u in (U - V) for v in V)

    part1 = sum1 / (union_size * len(U)) if len(U) > 0 else 0
    part2 = sum2 / (union_size * len(V)) if len(V) > 0 else 0

    return part1 + part2

def chord_metrics(pop: Population, annotation:list[tuple]) -> float:
    return chord_metrics_full_piece(pop, {0: annotation})

def chord_metrics_full_piece(pop: Population, annotations:dict[int,list[tuple]]) -> float:
    s = []
    interval = []
    lp_full = []
    lp_12 = []
    ah = []
    ahG = []
    p = []
    f = []
    fG = []
    w = []
    time_onsets = list(annotations.keys())
    for i, onset in enumerate(pop.archive):
        individual = pop.archive[onset].individual
        time_onset = time_onsets[i]
        extracted_features = extract_pitches(individual)
        annotated_features = [int(annotation[1]) for annotation in annotations[time_onset]]
        s.append(spiral_distance(extracted_features, annotated_features))
        interval.append(intervalic_distance(extracted_features, annotated_features))
        lp_full.append(lp_distance_all(extracted_features, annotated_features))
        lp_12.append(lp_distance_12(extracted_features, annotated_features))
        ah.append(avg_hausdorff_p(extracted_features, annotated_features))
        ahG.append(avg_hausdorff_p_G(extracted_features, annotated_features))
        p.append(projection_distance_general(extracted_features, annotated_features))
        f.append(fujita_distance(extracted_features, annotated_features))
        fG.append(fujita_distance_G(extracted_features, annotated_features))
        w.append(wasserstein(extracted_features, annotated_features))
    return np.mean(s), np.mean(interval), np.mean(lp_full), np.mean(lp_12), np.mean(ah), np.mean(ahG), np.mean(p), np.mean(f), np.mean(fG), np.mean(w)