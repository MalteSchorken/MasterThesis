import numpy as np

from .individual import BaseIndividual
from .sample_library import SampleLibrary

# Defaults from Vatolkin et. al (2020)
CHOOSE_MUTATION_P = [0.2, 0.4, 0.4]
SAMPLE_NUMBER_INCREASE_P = [1, 0.8, 0.4, 0.1, 0] # for 1, 2, 3, 4 or 5 samples currently present
ALPHA = 6
BETA = 3
L_BOUND = 1
U_BOUND = 10
PITCH_SHIFT_STD = 15

class Mutator:
    def __init__(self, 
                 sample_library:SampleLibrary, 
                 alpha:int=ALPHA, beta:int=BETA, 
                 l_bound:int=L_BOUND, u_bound:int=U_BOUND, 
                 sample_number_increase_p:list[float]=SAMPLE_NUMBER_INCREASE_P, 
                 pitch_shift_std:float=PITCH_SHIFT_STD, 
                 choose_mutation_p:list[float]=CHOOSE_MUTATION_P,
                 rng: np.random.Generator | None = None):
        """Creates an instance of the Mutator class.

        Parameters
        ----------
        sample_library : SampleLibrary
            Initialized sample library
        alpha : int, optional
            Used in calculation of number of mutations, by default ALPHA
        beta : int, optional
            Used in calculation of number of mutations, by default BETA
        l_bound : int, optional
            Minimum number of applied mutations to an individual, by default L_BOUND
        u_bound : int, optional
            Maximum number of applied mutations to an individual, by default U_BOUND
        sample_number_increase_p : list[float], optional
            Probabilites of increasing the number of samples if 
            the mutate_n_samples mutation is chosen, by default SAMPLE_NUMBER_INCREASE_P
        pitch_shift_std : float, optional
            Standard deviation of a pitch shift mutation. by default PITCH_SHIFT_STD
        choose_mutation_p : list[float], optional
            Probabilities of each mutation to be applied, by default CHOOSE_MUTATION_P
        """
        self.sample_library = sample_library
        self.alpha = alpha
        self.beta = beta
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.sample_number_increase_p = sample_number_increase_p
        self.pitch_shift_std = pitch_shift_std
        self.choose_mutation_p = choose_mutation_p
        self.rng = rng #if rng is not None else np.random.default_rng()

    def mutate_individual(self, individual:BaseIndividual) -> BaseIndividual:
        """Mutates an individual with one or more of the available mutation operations.

        Parameters
        ----------
        individual : BaseIndividual
            Individual that shall be mutated. 
            Note that python uses references and if you wish to 
            preserve the original individual, then pass a copy of it to this method instead.

        Returns
        -------
        BaseIndividual
            Mutated individual.
        """
        # Draw number of mutation
        n_mutations = int(np.clip(np.floor(self.rng.normal(loc=0, scale=1) * self.alpha + self.beta), a_min=self.l_bound, a_max=self.u_bound))
        for _ in range(n_mutations):
            # Decide which mutation to apply
            mutation = self.rng.choice([self.mutate_n_samples, self.mutate_instrument, self.mutate_pitch], p=self.choose_mutation_p)
            # Apply mutation
            mutated_individual = mutation(individual)
            # Set recalc fitness flag
            mutated_individual.recalc_fitness = True
            #mutated_individual.abs_stft = None
        
        return mutated_individual

    def mutate_n_samples(self, individual:BaseIndividual) -> BaseIndividual:
        """Mutates the number of samples in the individual.

        Parameters
        ----------
        individual : BaseIndividual
            Individual that shall be mutated.

        Returns
        -------
        BaseIndividual
            The individual after the number of samples was changed.
        """
        pre_mutation_n_samples = len(individual.samples)
        increase_probability = self.sample_number_increase_p[pre_mutation_n_samples - 1]
        # Increase or decrease number of samples
        rnd = self.rng.random()
        if rnd < increase_probability:
            # Add a sample
            new_sample = self.sample_library.get_random_sample_uniform(self.rng)
            individual.samples.append(new_sample)
        else:
            # Remove a sample
            idx = self.rng.choice(pre_mutation_n_samples)
            individual.samples.pop(idx)
        return individual

    def mutate_instrument(self, individual:BaseIndividual) -> BaseIndividual:
        """Changes one of the instruments in the individual to a different one.

        Parameters
        ----------
        individual : BaseIndividual
            Individual that shall be mutated.

        Returns
        -------
        BaseIndividual
            The individual after an instrument was changed.

        Raises
        ------
        RuntimeError
            If no new sample could be found for some reason.
        """
        pre_mutation_n_samples = len(individual.samples)

        # Choose one instrument
        change_idx = self.rng.choice(pre_mutation_n_samples)
        pitch = individual.samples[change_idx].pitch

        # Randomly change instrument and style uniformly
        new_instrument, new_style = self.sample_library.get_random_instrument_for_pitch(pitch, self.rng)
        #new_style = self.sample_library.get_random_style_for_instrument(new_instrument)

        # See if old pitch exists for new instrument
        new_sample = self.sample_library.get_sample(new_instrument, new_style, pitch)
        if new_sample:
            individual.samples[change_idx] = new_sample
        else:
            # Something went wrong
            raise RuntimeError("A sample for the requested instrument, style and pitch does not exist.")

        return individual

    def mutate_pitch(self, individual:BaseIndividual) -> BaseIndividual:
        """Changes the pitch of one of the samples in the given individual

        Parameters
        ----------
        individual : BaseIndividual
            Individual that shall be mutated.

        Returns
        -------
        BaseIndividual
            The individual after one of its samples' pitch was changed.
        """
        pre_mutation_n_samples = len(individual.samples)

        # Choose one instrument
        change_idx = self.rng.choice(pre_mutation_n_samples)
        chosen_sample = individual.samples[change_idx]
        
        # Choose a new pitch
        #new_pitch = self.sample_library.get_random_pitch_for_instrument_uniform(chosen_sample.instrument, chosen_sample.style)
        shift_by = np.floor(self.rng.normal(loc=0, scale=self.pitch_shift_std))
        new_pitch = self.sample_library.get_shifted_pitch(chosen_sample.instrument, chosen_sample.style, chosen_sample.pitch, shift_by)
        individual.samples[change_idx] = self.sample_library.get_sample(chosen_sample.instrument, chosen_sample.style, new_pitch)

        return individual
    
    def step_size_control(self, zeta:float):
        """
        Decrease std, mean and upper bound of the number of mutations applied 
        by multiplying each by zeta.

        Parameters
        ----------
        zeta: float
            Step size control parameter 
        """
        self.alpha *= zeta
        self.beta *= zeta
        self.u_bound = np.clip(self.u_bound * zeta, a_min=self.l_bound, a_max=None)