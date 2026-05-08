# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:45:52 2025

@author: malte
"""

import pickle
import os

from evoaudio.base_algorithms import approximate_piece
from evoaudio.sample_library import SampleLibrary
from evoaudio.mutations import Mutator
from evoaudio.individual import BaseIndividual

# Experiment parameters
RESULTS_PATH = "./experiments/moea_results_singleprocess.pkl"

N_RUNS = 100
MAX_STEPS = 5000
POPSIZE = 10
N_OFFSPRING = 1
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 10
ZETA = 0.9954


def run_experiment(target_individuals, sample_lib):
    """
    Run the approximation for a list of target individuals.
    Stores only final Population objects.
    """
    results = []

    for true_id, true_ind in target_individuals:
        print(f"Run {true_id + 1}/{len(target_individuals)}: {true_ind}")

        mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND)

        final_population = approximate_piece(
            target_y=true_ind.to_mixdown(),
            max_steps=MAX_STEPS,
            sample_lib=sample_lib,
            popsize=POPSIZE,
            n_offspring=N_OFFSPRING,
            onset_frac=1,
            zeta=ZETA,
            early_stopping_fitness=None,
            mutator=mutator,
            onsets=[0],
            verbose=True,
            parent_selection="random",
            survivor_selection="worst",
            objectives=[("stft", "kl_divergence"), ("stft", "js_divergence"), ("stft", "cosh")]
        )

        results.append((true_id, final_population))
        print(f"Finished Run {true_id + 1}/{len(target_individuals)}\n{'-' * 40}")

    return results


if __name__ == "__main__":
    # Load target individuals
    file_path = os.path.join("experiments", "ground_truth_scan", "ground_truth_dataset.pkl")
    with open(file_path, "rb") as f:
        restored_individuals = pickle.load(f)

    restored_with_ids = [(idx, ind) for idx, ind in enumerate(restored_individuals)]

    # Initialize sample library
    sample_lib = SampleLibrary()

    # Run the experiment
    result_logs = run_experiment(restored_with_ids, sample_lib)

    # Save final populations
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(result_logs, f)

    print(f"Experiment finished. Final populations saved to {RESULTS_PATH}")

