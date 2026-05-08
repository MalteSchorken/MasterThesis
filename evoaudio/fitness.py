import numpy as np
import librosa
from scipy.spatial.distance import minkowski, chebyshev, cosine
from scipy.stats import wasserstein_distance
from .individual import BaseIndividual

# ----------------------
# Distance metrics
# ----------------------
def cosh(x, y):
    return np.sum(x / y - np.log(x / y) + y / x - np.log(y / x) - 2) / len(x)

def wasserstein(x, y):
    return wasserstein_distance(x, y)

def itakura_saito(x, y):
    return np.sum((x / y) - np.log(x / y) - 1) / len(x)

def beta_divergence(x, y, beta=1):
    if beta == 1:
        return np.sum(x * (np.log(x / y)) - x + y)
    elif beta == 0:
        return np.sum(x / y - np.log(x / y) - 1)
    else:
        return np.sum((1 / (beta * (beta - 1))) * (x**beta + (beta - 1) * y**beta - beta * x * y**(beta - 1)))

def kullback_leibler(x, y):
    x = x / np.sum(x)
    y = y / np.sum(y)
    return np.sum(x * np.log(x / y))

def jensen_shannon(x, y):
    x = x / np.sum(x)
    y = y / np.sum(y)
    m = 0.5 * (x + y)
    return 0.5 * (np.sum(x * np.log(x / m)) + np.sum(y * np.log(y / m)))

def euclidean_distance(x, y):
    return minkowski(x, y, 2)

def manhattan_distance(x, y):
    return minkowski(x, y, 1)

def chebyshev_distance(x, y):
    return chebyshev(x, y)

def cosine_distance(x, y):
    return cosine(x, y)

def compute_stft(y, sr=22050, **kwargs):
    return np.abs(librosa.stft(y, **kwargs))

def compute_melspectrogram(y, sr=22050, **kwargs):
    return np.abs(librosa.feature.melspectrogram(y=y, sr=sr, **kwargs))

def compute_mfcc(y, sr=22050, **kwargs):
    return np.abs(librosa.feature.mfcc(y=y, sr=sr, **kwargs))

def compute_cqt(y, sr=22050, **kwargs):
    return np.abs(librosa.cqt(y=y, sr=sr, **kwargs))

def compute_pseudo_cqt(y, sr=22050, **kwargs):
    return np.abs(librosa.pseudo_cqt(y=y, sr=sr, **kwargs))

def compute_hybrid_cqt(y, sr=22050, **kwargs):
    return np.abs(librosa.hybrid_cqt(y=y, sr=sr, **kwargs))

def compute_vqt(y, sr=22050, **kwargs):
    return np.abs(librosa.vqt(y=y, sr=sr, **kwargs))

def compute_rms(y, **kwargs):
    return librosa.feature.rms(y=y, **kwargs)

def compute_iirt(y, **kwargs):
    return np.abs(librosa.iirt(y=y, **kwargs))

def compute_fmt(y, **kwargs):
    return np.abs(librosa.fmt(y=y, **kwargs))

def compute_tonnetz(y, sr=22050, **kwargs):
    return np.abs(librosa.feature.tonnetz(y=y, sr=sr, **kwargs))

def compute_spectral_combo(y, sr=22050, **kwargs):
    S = np.abs(librosa.stft(y, **kwargs))  # shared spectrogram base
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    return np.vstack([centroid, bandwidth, contrast, flatness, rolloff])

# Map keywords to functions
DISTANCE_FUNCTIONS = {
    "cosh": cosh,
    "itakura_saito": itakura_saito,
    "beta_divergence": beta_divergence,
    "kl_divergence": kullback_leibler,
    "js_divergence": jensen_shannon,
    "manhattan": manhattan_distance,  
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance,            
    "wasserstein": wasserstein_distance,
    "cosine": cosine_distance
}

AUDIO_FEATURE_FUNCTIONS = {
    "stft": compute_stft,
    "melspectrogram": compute_melspectrogram,
    "mfcc": compute_mfcc,
    "cqt": compute_cqt,
    "pseudo_cqt": compute_pseudo_cqt,
    "hybrid_cqt": compute_hybrid_cqt,
    "vqt": compute_vqt,
    "rms": compute_rms,
    "iirt": compute_iirt,
    "fmt": compute_fmt,
    "tonnetz": compute_tonnetz,
    "spectral": compute_spectral_combo
    # add more here
}

# ----------------------
# Generalized wrapper
# ----------------------
def distance_no_abs(abs_stft_x, abs_stft_y, metric="cosh", **kwargs):
    if metric not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(DISTANCE_FUNCTIONS.keys())}")
    a = np.average(abs_stft_x, axis=1)
    b = np.average(abs_stft_y, axis=1)
    return DISTANCE_FUNCTIONS[metric](a, b, **kwargs) if "beta_divergence" in metric else DISTANCE_FUNCTIONS[metric](a, b)

def compute_audio_feature(y, feature_name="stft", **kwargs):
    if feature_name not in AUDIO_FEATURE_FUNCTIONS:
        raise ValueError(f"Unknown feature '{feature_name}'. Available: {list(AUDIO_FEATURE_FUNCTIONS.keys())}")
    return AUDIO_FEATURE_FUNCTIONS[feature_name](y, **kwargs)

# ----------------------
# Fitness functions
# ----------------------


def multi_onset_fitness_cached(target, individual, objectives=[("stft", "cosh")], **kwargs):

    if individual.recalc_fitness:
        all_fitnesses = []

        # Render once
        y = individual.to_mixdown()

        # Precompute sample features for ALL objective feature names
        feature_names = list({obj[0] for obj in objectives})
        sample_features = {
            name: compute_audio_feature(y, feature_name=name)
            for name in feature_names
        }

        # Compare for each onset
        for onset in target.onsets:
            fitness_vector = []

            for feature_name, metric in objectives:
                sample_feature = sample_features[feature_name]
                target_feature = target.feature_per_snippet[onset][feature_name]

                val = distance_no_abs(sample_feature, target_feature, metric=metric, **kwargs)
                fitness_vector.append(val)

            all_fitnesses.append(fitness_vector)

        individual.fitness_per_onset = np.array(all_fitnesses)
        individual.recalc_fitness = False

    return individual.fitness_per_onset

def fitness(x, y) -> float:
    stft_x = librosa.stft(x)
    stft_y = librosa.stft(y)
    a = np.average(abs(stft_x), axis=1)
    b = np.average(abs(stft_y), axis=1)
    return cosh(a, b)