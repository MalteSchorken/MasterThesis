from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from glob import glob
import pickle
import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from evoaudio.base_algorithms import approximate_piece, _init_population
from evoaudio.base_sample import BaseSample
from evoaudio.individual import BaseIndividual
from evoaudio.mutations import Mutator
from evoaudio.population import Population, ArchiveRecord
from evoaudio.sample_library import SampleLibrary
from evoaudio.target import Target
from evoaudio.jaccard import calc_and_save_jaccard, calc_jaccard_for_chord_approximation
from evoaudio.chord_metrics import chord_metrics
from evoaudio.instrument_metric import avg_hausdorff_instruments
from evoaudio.log_processing import process_diversity_logs, process_error_logs, process_fitness_logs, process_instrument_logs, process_pitch_logs

RESULT_CSV = "./experiments/moea.csv"

N_RUNS = 100
MAX_STEPS = 5000
POPSIZE = 10
N_OFFSPRING = 1
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 10
ZETA = 0.9954
MAX_PROCESSES = 25

class LibraryManager(BaseManager):
    pass

class Logger():
    def __init__(self) -> None:
        self.logged_instrument_distances = []
        self.logged_pitch_distances = []
        self.logged_errors = []
        self.logged_fitnesses = []
        self.logged_diversity = []
    def log_errors(self, pop, annotation):
        errors = calc_jaccard_for_chord_approximation(pop=pop, annotation=annotation)
        self.logged_errors.append(errors)
    def log_fitness(self, pop):
        self.logged_fitnesses.append(pop.archive[0].fitness_vec)
    def log_diversity(self, pop):
        unique = set(pop.individuals)  # relies on __hash__ and __eq__ in BaseIndividual
        self.logged_diversity.append(len(unique))
    def log_instrument_distance(self, pop, annotation):
        errors = avg_hausdorff_instruments(pop=pop, annotation=annotation)
        self.logged_instrument_distances.append(errors)
    def log_pitch_distance(self, pop, annotation):
        errors = chord_metrics(pop=pop, annotation=annotation)
        self.logged_pitch_distances.append(errors)
    
def create_sample_set(sample_lib:SampleLibrary):
    individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, phi=1.0) for _ in range(N_RUNS)]
    annotations = [[(sample.instrument, str(sample.pitch.value)) for sample in ind.samples] for ind in individuals]
    return individuals, annotations

def objectives_to_name(objectives):
    return "__".join(f"{feat}_{metric}" for feat, metric in objectives)

distance_measures = [
         "itakura_saito",
         "beta_divergence",
         "cosh",
         "kl_divergence",
         "js_divergence",
         "manhattan",
         "euclidean",
         "chebyshev",
         "wasserstein",
         "cosine"
     ]
audio_features = [
         "stft",
         "melspectrogram",
         "mfcc",
         "cqt",
         "pseudo_cqt",
         "hybrid_cqt",
         "vqt",
         "rms",
         "iirt",
         "fmt",
         "tonnetz",
         "spectral"
     ]

def parse_objective(obj_str, audio_features, distance_measures):
    """
    Converts 'iirt_beta_divergence' -> ('iirt', 'beta_divergence')
    """
    for af in audio_features:
        prefix = af + "_"
        if obj_str.startswith(prefix):
            dm = obj_str[len(prefix):]
            if dm in distance_measures:
                return (af, dm)

    raise ValueError(f"Could not parse objective string: {obj_str}")
    
def sample_fitness_functions(df, n, random_state):
    if n > len(df):
        raise ValueError("n is larger than available fitness functions.")
    return df.sample(n=n, random_state=random_state)

def sample_objective_set(df, n, random_state):
    sampled = sample_fitness_functions(df, n, random_state)["objectives"]
    return [
        parse_objective(obj, audio_features, distance_measures)
        for obj in sampled
    ]


def run_experiment(target_individuals: list[tuple[int, BaseIndividual]], sample_lib:SampleLibrary,
                   error_logs:list, fitness_logs:list, diversity_logs:list, instrument_logs:list, pitch_logs:list,
                   result_logs:list, proc_id:int, objectives: list[tuple[str, str]], seed:int):
    
    
    
    def callback(pop:Population, step:int):
        annotation = [(sample.instrument, sample.pitch) for sample in true_ind.samples]
        logger.log_errors(pop=pop, annotation=annotation)
        logger.log_fitness(pop=pop)
        logger.log_diversity(pop=pop)
        logger.log_instrument_distance(pop=pop, annotation=annotation)
        logger.log_pitch_distance(pop=pop, annotation=annotation)

    for true_id, true_ind in target_individuals:
        rng = np.random.default_rng(seed + 1000*true_id)
        print("Searching for: " + str(true_ind))
        logger = Logger()
        mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND, rng=rng) 
        result = approximate_piece(
            target_y=true_ind.to_mixdown(), max_steps=MAX_STEPS, 
            sample_lib=sample_lib, popsize=POPSIZE, 
            n_offspring=N_OFFSPRING, onset_frac=1, 
            zeta=ZETA, early_stopping_fitness=None, 
            mutator=mutator, onsets=[0], verbose=proc_id==0, callback=callback,
            parent_selection="random", survivor_selection="worst",
            objectives = objectives, population = None, rng=rng,
            seed = seed+1000*true_id, algorithm = "smsemoa")


        error_logs.append(logger.logged_errors)
        fitness_logs.append(logger.logged_fitnesses)
        diversity_logs.append(logger.logged_diversity)
        instrument_logs.append(logger.logged_instrument_distances)
        pitch_logs.append(logger.logged_pitch_distances)
        result._flatten()
        result_logs.append((true_id, result))
        print("-----------------------------------")
        

if __name__ == "__main__":
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_id", type=int, required=True)
        args = parser.parse_args()
        
        run_id = args.run_id
        print(f"Running experiment {run_id}")
            
        file_path = os.path.join("experiments", "ground_truth_scan", "ground_truth_dataset.pkl")
        with open(file_path, "rb") as f:
            restored_individuals = pickle.load(f)
        restored_with_ids = [(idx, ind) for idx, ind in enumerate(restored_individuals)]
        
        recall_data = pd.read_csv("success_rate.csv")
        df_ea = recall_data[recall_data["algorithm"] == "EA"].copy()
        top_50_ea = (
            df_ea
            .sort_values("recall", ascending=False)
            .head(50)
            .reset_index(drop=True)
        )
        
        LibraryManager.register('SampleLibrary', SampleLibrary)
        with LibraryManager() as manager:
            shared_lib = manager.SampleLibrary()

            objective_sets = [
                sample_objective_set(df = top_50_ea, n = 2, random_state = run_id),
                sample_objective_set(df = top_50_ea, n = 3, random_state = run_id),
                sample_objective_set(df = top_50_ea, n = 4, random_state = run_id),
                sample_objective_set(df = top_50_ea, n = 5, random_state = run_id),
            ]

            save_path = os.path.join("experiments", "moea", "smsemoa", "shapley_recall")
            os.makedirs(save_path, exist_ok=True)

            for objectives in objective_sets:

                print(f"\n=== Running objectives: {objectives} ===")

                # Pretty name for files
                obj_name = objectives_to_name(objectives)

                # Shared logs
                manager_2 = Manager()
                error_logs = manager_2.list()
                fitness_logs = manager_2.list()
                diversity_logs = manager_2.list()
                instrument_logs = manager_2.list()
                pitch_logs = manager_2.list()
                result_logs = manager_2.list()

                # Run experiments
                ind_per_process = int(len(restored_individuals) / MAX_PROCESSES)

                processes = [
                    Process(
                        target=run_experiment,
                        args=(
                            restored_with_ids[proc_id * ind_per_process:(proc_id + 1) * ind_per_process],
                            shared_lib,
                            error_logs,
                            fitness_logs,
                            diversity_logs,
                            instrument_logs,
                            pitch_logs,
                            result_logs,
                            proc_id,
                            objectives,
                            run_id
                        )
                    )
                    for proc_id in range(MAX_PROCESSES)
                ]

                for process in processes:
                    process.start()
                for process in processes:
                    process.join()

                # Aggregate logs
                j_i_mean, j_p_mean, j_ip_mean = process_error_logs(error_logs, MAX_STEPS)
                instrument_mean = process_instrument_logs(instrument_logs, MAX_STEPS)
                pitch_means = process_pitch_logs(pitch_logs, MAX_STEPS)
                diversity_mean = process_diversity_logs(diversity_logs, MAX_STEPS)
                fitness_mean = process_fitness_logs(fitness_logs, MAX_STEPS)

                # Split multi-objective fitness into named columns
                fitness_cols = {
                    f"fitness_{feat}_{metric}": fitness_mean[:, i]
                    for i, (feat, metric) in enumerate(objectives)
                }

                # Final dataframe
                results = {
                    "j_i_mean": j_i_mean,
                    "j_p_mean": j_p_mean,
                    "j_ip_mean": j_ip_mean,
                    "diversity_mean": diversity_mean,
                    "instrument_mean": instrument_mean,
                    **pitch_means,
                    **fitness_cols,
                }

                df = pd.DataFrame(results)

                # Save
                csv_path = os.path.join(save_path, f"{run_id}__{obj_name}.csv")
                pop_path = os.path.join(save_path, f"{run_id}__{obj_name}_populations.pkl")

                df.to_csv(csv_path, index=False)
                with open(pop_path, "wb") as f:
                    pickle.dump(list(result_logs), f)

                print(f"Saved results to:\n  {csv_path}\n  {pop_path}")
