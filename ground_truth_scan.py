from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from glob import glob
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from evoaudio.base_algorithms import approximate_piece
from evoaudio.base_sample import BaseSample
from evoaudio.fitness import fitness_cached
from evoaudio.individual import BaseIndividual
from evoaudio.mutations import Mutator
from evoaudio.population import Population, ArchiveRecord
from evoaudio.sample_library import SampleLibrary
from evoaudio.target import Target
from evoaudio.jaccard import calc_and_save_jaccard, calc_jaccard_for_chord_approximation
from evoaudio.chord_metrics import chord_metrics
from evoaudio.instrument_metric import avg_hausdorff_instruments
from evoaudio.log_processing import process_diversity_logs, process_error_logs, process_fitness_logs, process_instrument_logs, process_pitch_logs

RESULT_CSV = "./experiments/ground_truth_scan.csv"

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
        self.logged_fitnesses.append(pop.archive[0].fitness)
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

def run_experiment(target_individuals: list[tuple[int, BaseIndividual]], sample_lib:SampleLibrary,
                   error_logs:list, fitness_logs:list, diversity_logs:list, instrument_logs:list, pitch_logs:list,
                   result_logs:list, proc_id:int, audio_feature:str, distance:str):
    
    def callback(pop:Population, step:int):
        annotation = [(sample.instrument, sample.pitch) for sample in true_ind.samples]
        logger.log_errors(pop=pop, annotation=annotation)
        logger.log_fitness(pop=pop)
        logger.log_diversity(pop=pop)
        logger.log_instrument_distance(pop=pop, annotation=annotation)
        logger.log_pitch_distance(pop=pop, annotation=annotation)

    for true_id, true_ind in target_individuals:
        print("Searching for: " + str(true_ind))
        logger = Logger()
        mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND) 
        result = approximate_piece(
            target_y=true_ind.to_mixdown(), max_steps=MAX_STEPS, 
            sample_lib=sample_lib, popsize=POPSIZE, 
            n_offspring=N_OFFSPRING, onset_frac=1, 
            zeta=ZETA, early_stopping_fitness=None, 
            mutator=mutator, onsets=[0], verbose=False, callback=callback,
            parent_selection="random", survivor_selection="worst",
            distance = distance, audio_feature = audio_feature)
        error_logs.append(logger.logged_errors)
        fitness_logs.append(logger.logged_fitnesses)
        diversity_logs.append(logger.logged_diversity)
        instrument_logs.append(logger.logged_instrument_distances)
        pitch_logs.append(logger.logged_pitch_distances)
        result_logs.append((true_id, result))
        print("-----------------------------------")
        

if __name__ == "__main__":
        file_path = os.path.join("experiments", "ground_truth_scan", "ground_truth_drums_dataset.pkl")
        with open(file_path, "rb") as f:
            restored_individuals = pickle.load(f)
        restored_with_ids = [(idx, ind) for idx, ind in enumerate(restored_individuals)]
        
        LibraryManager.register('SampleLibrary', SampleLibrary)
        with LibraryManager() as manager:
            shared_lib = manager.SampleLibrary()
# =============================================================================
#             distance_measures = [
#                 "itakura_saito",
#                 "beta_divergence",
#                 "cosh",
#                 "kl_divergence",
#                 "js_divergence",
#                 "manhattan",
#                 "euclidean",
#                 "chebyshev",
#                 "wasserstein",
#                 "cosine"
#             ]
#             audio_features = [
#                 "stft",
#                 "melspectrogram",
#                 "mfcc",
#                 "cqt",
#                 "pseudo_cqt",
#                 "hybrid_cqt",
#                 "vqt",
#                 "rms",
#                 "iirt",
#                 "fmt",
#                 "tonnetz",
#                 "spectral"
#             ]
# =============================================================================
            distance_measures = [
                "itakura_saito",
                "beta_divergence",
                "cosh",
                "kl_divergence",
                "js_divergence",
                "manhattan",
                "euclidean",
                "chebyshev",
                "cosine"
            ]
            audio_features = [
                "stft",
                "cqt",
                "pseudo_cqt",
                "hybrid_cqt",
                "iirt",
            ]
            for audio_feature in audio_features:
                for distance in distance_measures:
                    # Create sample set
                    manager_2 = Manager()
                    error_logs = manager_2.list()
                    fitness_logs = manager_2.list()
                    diversity_logs = manager_2.list()
                    instrument_logs = manager_2.list()
                    pitch_logs = manager_2.list()
                    result_logs = manager_2.list()  
                    # Run experiments
                    ind_per_process = int(len(restored_individuals) / MAX_PROCESSES)
                    processes = [Process(target=run_experiment, 
                                        args=(restored_with_ids[proc_id*ind_per_process:(proc_id+1)*ind_per_process], 
                                            shared_lib, error_logs, fitness_logs, diversity_logs, instrument_logs,
                                            pitch_logs, result_logs, proc_id, audio_feature, distance)) for proc_id in range(MAX_PROCESSES)]
                    for process in processes:
                        process.start()
                    for process in processes:
                        process.join()
                        
                    j_i_mean, j_p_mean, j_ip_mean = process_error_logs(error_logs, MAX_STEPS)
                    instrument_mean = process_instrument_logs(instrument_logs, MAX_STEPS)
                    pitch_means = process_pitch_logs(pitch_logs, MAX_STEPS)
                    diversity_mean = process_diversity_logs(diversity_logs, MAX_STEPS)
                    fitness_mean = process_fitness_logs(fitness_logs, MAX_STEPS)

                    # Combine all results into a DataFrame
                    results = {
                        "j_i_mean": j_i_mean,
                        "j_p_mean": j_p_mean,
                        "j_ip_mean": j_ip_mean,
                        "fitness_mean": fitness_mean,
                        "diversity_mean": diversity_mean,
                        "instrument_mean": instrument_mean,
                        **pitch_means
                    }
                
                    df = pd.DataFrame(results)
                    
                    save_path = os.path.join("experiments", "ground_truth_scan")
                    os.makedirs(save_path, exist_ok=True)
                    file_name = f"{audio_feature}_{distance}.csv"
                    df.to_csv(os.path.join(save_path, file_name), index=False)
                    with open(os.path.join(save_path, f"{audio_feature}_{distance}_populations.pkl"), "wb") as f:
                        pickle.dump(list(result_logs), f)
         
