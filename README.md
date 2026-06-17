# Experiment Reproduction
- Download additional supplemental material from here: https://drive.google.com/drive/folders/1URvtT1r0kJlxDSG6PGnaWhvVuTzhyysX?usp=drive_link  
- Unzip and place the `audio` folder in the root directory. 
- Run the scripts MOEA_tests.py and MOEA_aam.py in the root directory to repeat my experiments.

# Environment

- Python 3.12.9
- Supported OS: Linux, Windows
- Install dependencies from `requirements.txt` as venv or with the python environment manager of your choosing.  

# Parameter Overview

## Population Parameters
**POPSIZE (µ)**: Number of individuals in the population  
**N_OFFSPRING (λ)**: Number of offspring per generation  
**ONSET_FRAC (φ)**: Fraction of best approximated onsets for individual fitness calculation  
**INITIAL_N_SAMPLES_P** = [0.1, 0.3, 0.3, 0.2, 0.1]: Probabilities for how many samples an individual uses per onset on initialisation.  
**MAX_STEPS**: Number of generations until termination  

## Mutation Parameters
**α** = 6: Used in calculation of number of mutations  
**β** = 3: Used in calculation of number of mutations  
**L_BOUND** = 1: Minimum number of mutations to an individual  
**U_BOUND** = 10: Maximum mumber of mutations to an individual  
The number of mutations applied to an individual is calculated as follows:
floor(G * α + β), where G is a Gaussian number with mean 0 and std=1. The result is clipped to the interval [L_BOUND, U_BOUND].  

**SAMPLE_NUMBER_INCREASE_P** = [1, 0.8, 0.4, 0.1, 0]: for 1, 2, 3, 4 or 5 samples currently present in the individual, determines the probability of an increase of samples, if the mutate_n_samples mutation is chosen.   

**CHOOSE_MUTATION_P** = [0.4, 0.4, 0.2]: Probabilities of each mutation to be applied.

Default values taken from [Vatolkin et. al. (2020)](https://ieeexplore.ieee.org/abstract/document/9185506)

## MOEA Parameters
**objectives** = [("stft", "cosh")]: Set of simultaneously optimised objectives. Each objective is defined as (*audio_feature*, *distance_metric*), optimized jointly in a multi-objective optimization framework.
- Audio features: stft, melspectrogram, mfcc, cqt, pseudo_cqt, hybrid_cqt, vqt, iirt, tonnetz, spectral
- Distance metrics: cosh, itakura_saito, beta_divergence, kl_divergence, js_divergence, manhattan, euclidean, chebyshev, wasserstein, cosine

**algorithm** = "smsemoa": Evolutionary multi-objective optimisation algorithm. Supported algorithms: EA, SMSEMOA, NSGA-II, NSGA-III.
