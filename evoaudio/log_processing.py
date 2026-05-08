# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 17:37:38 2025

@author: malte
"""
import numpy as np

def process_error_logs(error_logs, max_steps):
    j_i_logs, j_p_logs, j_ip_logs = [], [], []
    for log in error_logs:
        j_i, j_p, j_ip = [], [], []
        for tup in log:
            j_i.append(tup[0])
            j_p.append(tup[1])
            j_ip.append(tup[2])
        while len(j_i) < max_steps:
            j_i.append(0)
            j_p.append(0)
            j_ip.append(0)
        j_i_logs.append(j_i)
        j_p_logs.append(j_p)
        j_ip_logs.append(j_ip)

    return (
        np.mean(np.array(j_i_logs), axis=0),
        np.mean(np.array(j_p_logs), axis=0),
        np.mean(np.array(j_ip_logs), axis=0)
    )

def process_instrument_logs(instrument_logs, max_steps):
    i_logs = []
    for log in instrument_logs:
        instrument = list(log)
        while len(instrument) < max_steps:
            instrument.append(0)
        i_logs.append(instrument)
    return np.mean(np.array(i_logs), axis=0)

def process_pitch_logs(pitch_logs, max_steps):
    s_logs, int_logs = [], []
    lp_full_logs, lp_12_logs = [], []
    ah_logs, ahG_logs = [], []
    p_logs, f_logs = [], []
    fG_logs, w_logs = [], []

    for log in pitch_logs:
        s, i = [], []
        lp_full, lp_12 = [], []
        ah, ahG = [], []
        p, f = [], []
        fG, w = [], []
        
        for tup in log:
            s.append(tup[0])
            i.append(tup[1])
            lp_full.append(tup[2])
            lp_12.append(tup[3])
            ah.append(tup[4])
            ahG.append(tup[5])
            p.append(tup[6])
            f.append(tup[7])
            fG.append(tup[8])
            w.append(tup[9])

        while len(s) < max_steps:
            s.append(0)
            i.append(0)
            lp_full.append(0)
            lp_12.append(0)
            ah.append(0)
            ahG.append(0)
            p.append(0)
            f.append(0)
            fG.append(0)
            w.append(0)

        s_logs.append(s)
        int_logs.append(i)
        lp_full_logs.append(lp_full)
        lp_12_logs.append(lp_12)
        ah_logs.append(ah)
        ahG_logs.append(ahG)
        p_logs.append(p)
        f_logs.append(f)
        fG_logs.append(fG)
        w_logs.append(w)

    return {
        "s_mean": np.mean(np.array(s_logs), axis=0),
        "i_mean": np.mean(np.array(int_logs), axis=0),
        "lp_full_mean": np.mean(np.array(lp_full_logs), axis=0),
        "lp_12_mean": np.mean(np.array(lp_12_logs), axis=0),
        "ah_mean": np.mean(np.array(ah_logs), axis=0),
        "ahG_mean": np.mean(np.array(ahG_logs), axis=0),
        "p_mean": np.mean(np.array(p_logs), axis=0),
        "f_mean": np.mean(np.array(f_logs), axis=0),
        "fG_mean": np.mean(np.array(fG_logs), axis=0),
        "w_mean": np.mean(np.array(w_logs), axis=0)
    }

def process_diversity_logs(diversity_logs, max_steps):
# =============================================================================
#     diversity_array = np.array([
#         log + [np.nan] * (max_steps - len(log)) 
#         for log in diversity_logs
#     ])
#     diversity_mean = np.nanmean(diversity_array, axis=0)
#     return diversity_mean
# =============================================================================
    return 0

def process_fitness_logs(fitness_logs, max_steps):
    """
    fitness_logs: list of logs
        Each log = list of fitness vectors (one per generation)
        fitness vector = list[float] (one per objective)
    """
    for i, log in enumerate(fitness_logs):
        if len(log) == 0:
            print(f"[WARN] Run {i} has 0 fitness entries!")
        else:
            print(f"[OK] Run {i} length: {len(log)}, first entry: {log[0]}")
        
    fitness_logs = [log for log in fitness_logs if len(log) > 0]

    if len(fitness_logs) == 0:
        # No valid runs → return zeros
        return np.zeros((max_steps, 1))

    # --- 2) Determine number of objectives safely ---
    first_log = next((log for log in fitness_logs if len(log) > 0), None)

    if first_log is None or len(first_log[0]) == 0:
        # Something is structurally wrong
        raise ValueError("fitness_logs contain no valid fitness vectors.")

    n_objectives = len(first_log[0])

    # --- 3) Pad runs ---
    fitness_padded = []
    for log in fitness_logs:
        padded = [vec for vec in log]

        while len(padded) < max_steps:
            padded.append([0.0] * n_objectives)

        fitness_padded.append(padded)

    # (runs, steps, objectives)
    arr = np.array(fitness_padded)

    # Average over runs
    return np.mean(arr, axis=0)