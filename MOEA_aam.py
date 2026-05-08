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
from parsing.arff_parsing import parse_arff

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

POPSIZE = 300
N_OFFSPRING = 1
MAX_STEPS = 5000
ONSET_FRAC = 0.05
ALPHA = 5
BETA = 10
L_BOUND = 1
U_BOUND = 20
ZETA = 0.9954
MAX_PROCESSES = 20

class LibraryManager(BaseManager):
    pass

def create_sample_set():
    # Create sample set
    mixes = {file.split('_mix.mp3')[0][-4:]: librosa.load(file) for file in glob("./audio/tinyAAM/audio-mixes-mp3/*.mp3")}
    annotations = {file.split('_onsets.arff')[0][-4:]: parse_arff(file) for file in glob("./audio/tinyAAM/annotations/*onsets.arff")}
    return annotations, mixes

def get_valid_sample(sample_lib, instrument, pitch):
    # Retrying until valid style is found for pitch
    try:
        return sample_lib.get_sample(instrument=instrument, pitch=pitch)
    except:
        return get_valid_sample(sample_lib, instrument, pitch)

class Logger():
    def __init__(self) -> None:
        self.logged_fitnesses = []
    def log_fitness(self, pop):
        self.logged_fitnesses.append(pop.archive[0].fitness_vec)

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


def run_experiment(annotations, target_mixes, sample_lib:SampleLibrary,
                   fitness_logs:list, result_logs:list, proc_id:int, objectives: list[tuple[str, str]], seed:int):
    
    
    
    def callback(pop:Population, step:int):
        logger.log_fitness(pop=pop)

    for name in annotations:
        target_mix, target_sr = target_mixes[name]
        annotation_dict = annotations[name]
        onsets = [int(round(float(onset_time) * target_sr)) for onset_time in annotation_dict.keys()]
        #print(next(iter(annotation_dict.values())))
        #annotation_dict.values() = [(instr, int(pitch.replace("+",""))) for instr, pitch in annotation_dict.values()]
        for val in annotation_dict.values():
            for instr, pitch in val:
                pitch = int(pitch.replace("+", ""))
        rng = np.random.default_rng(seed)
        logger = Logger()
        mutator = Mutator(sample_library=sample_lib, alpha=ALPHA, beta=BETA, l_bound=L_BOUND, u_bound=U_BOUND, rng=rng) 
        result = approximate_piece(
            target_y=target_mix, max_steps=MAX_STEPS, 
            sample_lib=sample_lib, popsize=POPSIZE, 
            n_offspring=N_OFFSPRING, onset_frac=ONSET_FRAC, 
            zeta=ZETA, early_stopping_fitness=None, 
            mutator=mutator, onsets=onsets, verbose=proc_id==0, callback=callback,
            parent_selection="random", survivor_selection="worst",
            objectives = objectives, population = None, rng=rng,
            seed = seed, algorithm = "EA")


        fitness_logs.append(logger.logged_fitnesses)
        result._flatten()
        result_logs.append((name, annotation_dict, onsets, target_sr, result))
        print("-----------------------------------")
        

if __name__ == "__main__":
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_id", type=int, required=True)
        args = parser.parse_args()
        
        run_id = args.run_id
        print(f"Running experiment {run_id}")
            
        
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
            annotations, target_mixes = create_sample_set()

            objective_sets = [
                [("stft", "kl_divergence"), ("stft", "js_divergence")],
                [("stft", "kl_divergence"), ("stft", "js_divergence"), ("stft", "cosh")],
            ]

            # Create sample set
            save_path = os.path.join("experiments", "moea", "tiny_aam", "smsemoa", "recall")
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
                annotation_items = list(annotations.items())
                MAX_PROCESSES = min(MAX_PROCESSES, len(annotation_items))
                chunk_size = len(annotation_items) // MAX_PROCESSES
                chunks = [dict(annotation_items[i*chunk_size:(i+1)*chunk_size]) for i in range(MAX_PROCESSES)]

                processes = [
                    Process(
                        target=run_experiment,
                        args=(
                            chunks[proc_id],
                            target_mixes,
                            shared_lib,
                            fitness_logs,
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
                fitness_mean = process_fitness_logs(fitness_logs, MAX_STEPS)

                # Split multi-objective fitness into named columns
                fitness_cols = {
                    f"fitness_{feat}_{metric}": fitness_mean[:, i]
                    for i, (feat, metric) in enumerate(objectives)
                }

                # Final dataframe
                results = {
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
