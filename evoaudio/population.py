from __future__ import annotations
import bisect
import pickle
from typing import Union

import numpy as np

from .individual import BaseIndividual
from .base_sample import BaseSample, FlatSample

class Population:
    individuals: list[BaseIndividual]
    archive: dict[int, ArchiveRecord]
    
    def __init__(self) -> None:
        self.individuals = [] # List of BaseIndividuals, Sorted by fitness values
        self.archive = {} # Dict of onset: SampleCollection # TODO: Refactor and expand archive to hold records for each instrument
    
    def __str__(self) -> str:
        return "\n".join([str(individual) for individual in self.individuals])

    def init_archive(self, onsets:Union[list[int], np.ndarray[int]]) -> None:
        """Initializes the archive of best individuals per onset.

        Parameters
        ----------
        onsets : list[int]
            list of onsets from the target piece
        """
        for individual in self.individuals:
            for i, onset in enumerate(onsets):
                vec = individual.fitness_per_onset[i]
                score = float(np.sum(vec))
                
                if onset not in self.archive or score < self.archive[onset].score:
                    self.archive[onset] = ArchiveRecord(
                        onset=onset,
                        fitness_vec=vec,
                        individual=individual
                    )
                    
    def sort_individuals_by_fitness(self):
        """Sorts the list of individuals by fitness. Meant to only be done upon initialization. 
        """
        self.individuals.sort(key=lambda ind: sum(ind.fitness))

    def insert_individual(self, individual:BaseIndividual):
        """Inserts an individual into the population. 
        Insertion is done into the self.individuals list, preserving fitness order.

        Parameters
        ----------
        individual : BaseIndividual
            An individual containing one or more samples and calculated fitness value.
        """
        # Insert individual
        bisect.insort_left(self.individuals, individual, key=lambda item: sum(item.fitness))
        # Update record of best onset approximations
        for i, onset in enumerate(self.archive):
            vec = individual.fitness_per_onset[i]
            score = float(np.sum(vec))
            
            if score < self.archive[onset].score:
                self.archive[onset].fitness_vec = vec
                self.archive[onset].score = score
                self.archive[onset].individual = individual
                
    def remove_worst(self, n:int):
        """Removes the worst n individuals from the population.
        This method assumes that self.individuals is already sorted by fitness.

        Parameters
        ----------
        n : int
            Number of individuals to remove from the population.
        """
        self.individuals = self.individuals[:-n]

    def get_best_individual(self) -> BaseIndividual:
        """Returns the individual with highest fitness.

        Returns
        -------
        BaseIndividual
            Individual with the highest fitness
        """
        return self.individuals[0]
    
    def merge_populations(self, other_pop:Population):
        """Merges this population with another, taking into account mismatches in the approximated onsets.

        Parameters
        ----------
        other_pop : Population
            The other population to merge with.

        Returns
        -------
        bool
            True if merge was successful and no mismatch was detected. 
            False if an onset mismatch was detected. 
            In that case it is recommended to recalculate the 
            fitness for all individuals with recalc_fitness == True.
        """
        # Update records
        onset_mismatch = False
        for onset, record in other_pop.archive.items():
            if onset in self.archive:
                if record.fitness < self.archive[onset].fitness:
                    self.archive[onset] = record
            else: 
                self.archive[onset] = record
                onset_mismatch = True
        # Merge individual list
        for individual in other_pop.individuals:
            individual.recalc_fitness = True
        self.individuals += other_pop.individuals

        return not onset_mismatch
    
    def _flatten(self):
        """Flattens the population to reduce disk space usage.
        """
        for individual in self.individuals:
            for i, sample in enumerate(individual.samples):
                individual.samples[i] = FlatSample(sample.instrument, sample.style, sample.pitch)
        for record in self.archive.values():
            individual = record.individual
            for i, sample in enumerate(individual.samples):
                individual.samples[i] = FlatSample(sample.instrument, sample.style, sample.pitch)

    def _expand(self, sample_lib):
        """Expands a flattened population by reloading the included samples from the sample library.

        Parameters
        ----------
        sample_lib : SampleLibrary
            Initialized sample library from which to load the samples in this population.
        """
        for individual in self.individuals:
            for i, sample in enumerate(individual.samples):
                individual.samples[i] = sample_lib.get_sample(sample.instrument, sample.style, sample.pitch)
        for record in self.archive.values():
            individual = record.individual
            for i, sample in enumerate(individual.samples):
                individual.samples[i] = sample_lib.get_sample(sample.instrument, sample.style, sample.pitch)

    def save_as_file(self, filename:str, flatten:bool=True):
        """Saves the population to a pickled file.

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
        
    @classmethod 
    def from_file(cls, filename:str, expand:bool=True, sample_lib=None) -> Population:
        """Loads a population from a pickled file.

        Parameters
        ----------
        filename : str
            Name of the population file.
        expand : bool
            Expands the samples from FlatSamples back to BaseSamples, provided a sample_lib is given.
        sample_lib : SampleLibrary
            If expand, provide the SampleLibrary from which to load the expanded sample.
        Returns
        -------
        Population
            The loaded population contained in the given file.
        """
        with open(filename, 'rb') as fp:
            obj = pickle.load(fp)
            if expand and sample_lib is not None:
                obj._expand(sample_lib)
            
            return obj

class ArchiveRecord():
    def __init__(self, onset: int, fitness_vec=None, individual=None):
        self.onset = onset
        self.fitness_vec = fitness_vec  # vector of objectives
        self.score = None if fitness_vec is None else float(np.sum(fitness_vec))
        self.individual = individual