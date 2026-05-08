from copy import copy

import numpy as np
import librosa
import pickle

from .sample_library import SampleLibrary
from .base_sample import BaseSample
from evoaudio.pitch import Pitch

INITIAL_N_SAMPLES_P = [0.1, 0.3, 0.3, 0.2, 0.1]

class BaseIndividual:
    samples: list[BaseSample]
    phi: float
    fitness_per_onset: list[list[float]] | np.ndarray
    fitness: list[float] | np.ndarray
    recalc_fitness: bool
    #abs_stft: np.ndarray
    
    def __hash__(self):
        return hash(tuple((s.instrument, s.style, s.pitch) for s in self.samples))

    def __eq__(self, other):
        return isinstance(other, BaseIndividual) and all(
            (s1.instrument, s1.style, s1.pitch) == (s2.instrument, s2.style, s2.pitch)
            for s1, s2 in zip(self.samples, other.samples)
        )

    def __init__(self, phi:float=0.1):
        self.samples = [] # List of samples in the collection
        self.phi = phi # Fraction of onsets that form the basis of overall fitness for this individual 
        self.fitness_per_onset = [] # Vector of fitnesses per onset
        self.fitness = [] # Mean fitness to top φ% of approximated onsets
        self.recalc_fitness = True # True if sample has been modified but fitness has yet to be recalculated
    
    def __str__(self):
        s = f"Fitness: {self.fitness} | " + ", ".join(str(x) for x in self.samples)
        return s

    def calc_phi_fitness(self):
        """
        Multi-objective φ-fitness:
        For each objective column:
          - select the top φ fraction of onsets
          - compute their mean
        """
        fitness_matrix = np.array(self.fitness_per_onset)  # shape: (num_onsets, num_obj)
    
        num_onsets, num_obj = fitness_matrix.shape
        k = int(np.ceil(num_onsets * self.phi))
    
        final = []
    
        for j in range(num_obj):    # loop over objectives
            column = fitness_matrix[:, j]  # (num_onsets,)
            
            # indices of k smallest values in this column
            idx = np.argpartition(column, k - 1)[:k]
    
            # mean of the best ones
            Fj = column[idx].mean()
            final.append(Fj)
    
        self.fitness = np.array(final)
        self.recalc_fitness = False

    def to_mixdown(self) -> np.ndarray:
        """Creates a mix of the samples contained in the collection.

        Returns
        -------
        np.ndarray
            Mix of the samples.
        """
        # Resize by expanding all samples to the same length
        max_length = np.max([len(sample.y) for sample in self.samples])
        ys_equal_length = [np.pad(sample.y, (0, max_length - len(sample.y))) for sample in self.samples]
        return np.sum(ys_equal_length, axis=0)

    @classmethod
    def from_copy(cls, obj):
        """Efficiently creates a copy of the given Individual.

        Parameters
        ----------
        obj : BaseIndividual
            Individual that shall be copied.

        Returns
        -------
        BaseIndividual
            Equivalent copy of the Individual that can be modified without modifying the original.
        """
        instance = cls()
        instance.samples = [copy(sample) for sample in obj.samples] # Copies only the reference to a sample for better performance.
        instance.phi = obj.phi
        instance.fitness_per_onset = [fitness for fitness in obj.fitness_per_onset]
        instance.recalc_fitness = obj.recalc_fitness
        instance.fitness = obj.fitness
        return instance

    @classmethod
    def create_random_individual(cls, sample_lib:SampleLibrary, rng: np.random.Generator, max_samples:int=5, sample_num_p:list[float]=INITIAL_N_SAMPLES_P, phi:float=0.1):
        """Creates an individual from a sample library and given parameters.

        Parameters
        ----------
        sample_lib : SampleLibrary
            SampleLibrary object containing the instrument and pitch information,
            as well as the samples themselves.
        max_samples : int, optional
            Maximum number of samples in an individual upon initialization, by default 5.
        sample_num_p : list[float], optional
            List probabilities of the number of samples from 1 to max_samples, by default [0.1, 0.3, 0.3, 0.2, 0.1].
        phi : float, optional
            Fraction of onsets that affect fitness calculation, by default 0.1.

        Returns
        -------
        BaseIndividual
            Initialized BaseIndividual containing samples from the provided SampleLibrary.
        """
        individual = cls(phi=phi)
        n_samples = rng.choice(
            np.arange(1, max_samples + 1),
            p=sample_num_p
        )
        
        for _ in range(n_samples):
            rng.random()
            individual.samples.append(sample_lib.get_random_sample_uniform(rng))
        return individual
    
    def add_sample(self, sample:BaseSample):
        self.samples.append(sample)
    
    @classmethod
    def create_individual(cls, sample_lib:SampleLibrary, n_samples:int, instruments:list[str],
                          styles:list[str], pitches:list[Pitch], phi:float=0.1):
        individual = cls(phi=phi)
        for i in range(n_samples):
            individual.samples.append(sample_lib.get_sample(instrument = instruments[i], style = styles[i], pitch = pitches[i]))
        return individual
    
        
    def save_as_file(self, filename:str, flatten:bool=False):
        """Saves the individual to a pickled file.

        Parameters
        ----------
        filename : str
            Desired name of the file.
        flatten : bool 
            If True, will turn all samples into FlatSample to drastically reduce disk space.
            (Use expand=True when the .pkl file is read later)
        """
        if flatten:
            self._flatten()
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)
