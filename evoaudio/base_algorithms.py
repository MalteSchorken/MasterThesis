from typing import Any, Callable, Union

import numpy as np
from tqdm import tqdm

from .sample_library import SampleLibrary
from .individual import BaseIndividual
from .mutations import Mutator
from .fitness import multi_onset_fitness_cached
from .population import Population
from .population_logging import PopulationLogger
from .target import Target

def approximate_piece(target_y:Union[np.ndarray, list], max_steps:int, 
                      sample_lib:SampleLibrary, popsize:int, n_offspring:int, 
                      onset_frac:float, zeta:float=None, early_stopping_fitness:float=None, 
                      population:Population=None, mutator:Mutator=None, logger:PopulationLogger=None, 
                      onsets:Union[np.ndarray, list]=None, verbose:bool=True, callback:Callable[[Population, int], Any]=None,
                      parent_selection: str = "random", survivor_selection: str = "worst",
                      objectives: list[tuple[str, str]] = [("stft", "cosh")], 
                      rng: np.random.Generator | None = None, seed: int | None = None,
                      algorithm: str = "smsemoa"
                      ) -> Population:
    """Evolutionary approximation of a polyphonic musical piece.

    Parameters
    ----------
    target_y : Union[np.ndarray, list]
        Signal of the target musical piece, as imported by librosa.
    max_steps : int
        Maximum number of iterations (generations) before termination.
    sample_lib : SampleLibrary
        Library of samples which define the algorithm's search space.
    popsize : int
        Size of the population (µ).
    n_offspring : int
        Number of offspring (λ) per generation.
    onset_frac : float
        Fraction of approximated onsets (φ) per individual.
    zeta : float, optional
        Optional parameter for step size adaptation.
    early_stopping_fitness : float, optional
        Algorithm will terminate early if this value is provided and
        the best individual.
        achieves a fitness below this threshold.
    population : Population, optional
        Pre-initialized population object.
    mutator : Mutator, optional
        Pre-initialized Mutator object that supports the
        mutate_individual(BaseIndividual) method.
    logger : PopulationLogger, optional
        Logging object, if desired. Can be None to omit logging.
    onsets : Union[np.ndarray, list], optional
        Positions of onsets (in samples) within the target piece. 
        If not provided, they will be estimated by librosa.onset.onset_detect.
    verbose : bool, optional
        If True, will print a progress bar and additional information to
        console during each step.
    callback : Callable, optional
        Callback function that receives a population and the
        current step as input.
    parent_selection : str, optional
        Strategy used to select parents for reproduction.
        Available options are: "random", "tournament", or "roulette".
    survivor_selection : str, optional
        Strategy used to remove individuals from the population when using the
        standard EA replacement mechanism. Available options are: "worst",
        "tournament", "roulette", or "tournament_no_elite".
    objectives : list[tuple[str, str]], optional
        List of objective functions used for evaluating individuals. Each tuple
        specifies the feature representation and the distance metric used for
        computing the fitness, e.g. ("stft", "cosh"). The objectives define the
        dimensions of the multi-objective optimization problem.
    rng : np.random.Generator, optional
        NumPy random number generator used throughout the algorithm for all
        stochastic operations such as initialization, mutation, and selection.
        If not provided, a new generator will be created.
    seed : int, optional
        Random seed used to initialize the NumPy random number generator when
        `rng` is not provided.
    algorithm : str, optional
        Multi-objective evolutionary algorithm used for survivor selection.
        Supported options are: "EA", "smsemoa", "nsga2", "nsga3"
    Returns
    -------
    Population
        The full population of individual approximations after max_steps of iterations.
    """
    if rng is None:
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
    
    # Initialization
    if mutator is None:
        mutator = Mutator(sample_lib, rng = rng) # Applies mutations and handles stft updates
    target = Target(target_y, onsets, objectives)

    # Create initial population
    if population is None:
        population = _init_population(sample_lib=sample_lib, target=target, onset_frac=onset_frac, popsize=popsize,
                                      verbose=verbose, objectives=objectives, rng=rng)

    # Evolutionary Loop
    for step in (pbar := tqdm(range(max_steps), disable=(not verbose))):
        done = _step(population=population, target=target, n_offspring=n_offspring, mutator=mutator, zeta=zeta,
                     early_stopping_fitness=early_stopping_fitness, logger=logger, step=step,
                     parent_selection=parent_selection, survivor_selection=survivor_selection,
                     objectives = objectives, rng=rng, seed=seed, algorithm=algorithm)
        
        if verbose:
            N = len(population.individuals)
            # Update progress bar
            pbar.set_postfix_str(f"Best individual: {str(population.get_best_individual())}, Pop Size: {N}")
        if callback is not None:
            callback(population, step)
        # Early stopping
        if done:
            break

    # Return final population
    return population

def _init_population(sample_lib:SampleLibrary, target:Target, onset_frac:float, popsize:int, verbose:bool,
                     objectives: list[tuple[str, str]] = [("stft", "cosh")], rng: np.random.Generator | None = None) -> Population:
    # Create initial population
    if rng is None:
        rng = np.random.default_rng()
    population = Population()
    population.individuals = [BaseIndividual.create_random_individual(sample_lib=sample_lib, rng=rng, phi=onset_frac) for _ in tqdm(range(popsize), desc="Initializing Population", disable=(not verbose))]
    for individual in tqdm(population.individuals, desc="Calculating initial fitness", disable=(not verbose)):
        # Calc initial fitness
        individual.fitness_per_onset = multi_onset_fitness_cached(target, individual,
                                                                  objectives = objectives)
        individual.calc_phi_fitness()
    population.init_archive(target.onsets) # Initial record of best approximations of each onset
    population.sort_individuals_by_fitness() # Sort population for easier management
    return population


def _step(population:Population, target:Target, n_offspring:int, mutator:Mutator=None,
          zeta:float=None, early_stopping_fitness:float=None, logger:PopulationLogger=None,
          step:int=None, parent_selection: str = "random", survivor_selection: str = "worst",
          objectives: list[tuple[str, str]] = [("stft", "cosh")], rng: np.random.Generator | None = None,
          seed: int | None = None, algorithm: str = "smsemoa"):
    # Create lambda offspring
    parents = select_parents(population, n_parents=n_offspring, method=parent_selection, rng = rng)
    offspring = [mutator.mutate_individual(BaseIndividual.from_copy(individual)) for individual in parents]
    
    # Evaluate fitness of offspring
    for individual in offspring:
        individual.fitness_per_onset = multi_onset_fitness_cached(target, individual,
                                                                  objectives = objectives)
        individual.calc_phi_fitness()
        # Insert individual into population
        population.insert_individual(individual)
    # Remove lambda worst individuals
    match algorithm:
        case "EA":
            remove_individuals(population, n_remove=n_offspring, method=survivor_selection)
        case "smsemoa":
            remove_sms_emoa(population)
        case "nsga2":
            remove_nsga2(population, seed)  
        case "nsga3":
            remove_nsga3(population)
        case _:
            raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            "Expected one of: EA, smsemoa, nsga2, nsga3"
        )
    # Step size adaptation
    if zeta is not None:
        mutator.step_size_control(zeta)

    if logger is not None:
        logger.log_population(population, step)
    
    # Early stopping
    if (early_stopping_fitness is not None 
        and population.get_best_individual().fitness <= early_stopping_fitness):
        return True
    else:
        return False

def select_parents(population, n_parents, rng, method="random", k=3):
    if rng is None:
        rng = np.random.default_rng()
    individuals = population.individuals
    if method == "random":
        return rng.choice(individuals, size=n_parents)
    elif method == "tournament":
        return [min(rng.choice(individuals, size=k), key=lambda ind: ind.fitness) for _ in range(n_parents)]
    elif method == "roulette":
        fitnesses = np.array([1.0 / (ind.fitness**2 + 1e-8) for ind in individuals])  # Inverse fitness
        probs = fitnesses / np.sum(fitnesses)
        return rng.choice(individuals, size=n_parents, p=probs)
    else:
        raise ValueError(f"Unknown parent selection method: {method}")
        
def remove_individuals(population, n_remove, method="worst", tournament_k=3, elite_k=3):
    if method == "worst":
        population.remove_worst(n_remove)
    elif method == "tournament":
        for _ in range(n_remove):
            candidates = np.random.choice(population.individuals, size=tournament_k, replace=False)
            worst = max(candidates, key=lambda ind: ind.fitness)
            population.individuals.remove(worst)
    elif method == "roulette":
        fitnesses = np.array([(ind.fitness**2 + 1e-8) for ind in population.individuals])
        probs = fitnesses / np.sum(fitnesses)
        to_remove = np.random.choice(population.individuals, size=n_remove, replace=False, p=probs)
        for ind in to_remove:
            population.individuals.remove(ind)
    elif method == "tournament_no_elite":
        population.sort_individuals_by_fitness()
        # Protect the top-k elite
        non_elite_individuals = population.individuals[elite_k:]
        # Tournament on the non-elite individuals
        for _ in range(n_remove):
            competitors = np.random.choice(non_elite_individuals, size=tournament_k, replace=False)
            worst = max(competitors, key=lambda ind: ind.fitness)
            population.individuals.remove(worst)
    else:
        raise ValueError(f"Unknown survivor selection method: {method}")
                
from pymoo.core.population import Population as PymooPop
from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga2 import NSGA2

class NoConstraintProblem(Problem):
    def __init__(self, n_obj=1):
        super().__init__(n_var=1, n_obj=n_obj, n_constr=0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.zeros((len(x), self.n_obj))

def remove_nsga2(population, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = len(population.individuals)
    if N <= 1:
        return

    # number of objectives
    n_obj = len(population.individuals[0].fitness)

    # build pymoo population
    X = np.arange(N).reshape(N, 1)
    F = np.zeros((N, n_obj))
    pymoo_pop = PymooPop().new("X", X, "F", F)

    for i, ind in enumerate(population.individuals):
        pymoo_pop[i].F = np.asarray(ind.fitness, dtype=float)

    # NSGA-II algorithm (no reference directions needed)
    algo = NSGA2()
    problem = NoConstraintProblem()

    # select N-1 survivors
    survivors = algo.survival.do(
        problem,
        pymoo_pop,
        n_survive=N - 1
    )
    survivor_indices = [int(ind.X[0]) for ind in survivors]
    population.individuals = [population.individuals[i] for i in survivor_indices]

def remove_nsga3(population):

    N = len(population.individuals)
    if N <= 1:
        return

    # number of objectives
    n_obj = len(population.individuals[0].fitness)

    # pymoo population
    X = np.arange(N).reshape(N, 1)
    F = np.zeros((N, n_obj))
    pymoo_pop = PymooPop().new("X", X, "F", F)

    for i, ind in enumerate(population.individuals):
        pymoo_pop[i].F = np.array(ind.fitness, dtype=float)

    # reference directions for NSGA-III
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_points=N)

    # NSGA-III algorithm
    algo = NSGA3(ref_dirs=ref_dirs)
    problem = NoConstraintProblem()
    # selection of survivors
    survivors = algo.survival.do(
        problem,    
        pymoo_pop,
        n_survive=N-1
    )
    survivor_indices = [int(ind.X[0]) for ind in survivors]
    population.individuals = [population.individuals[i] for i in survivor_indices]

def remove_sms_emoa(population):
    N = len(population.individuals)
    if N <= 1:
        return
    # number of objectives
    n_obj = len(population.individuals[0].fitness)

    # pymoo population
    X = np.arange(N).reshape(N, 1)
    F = np.zeros((N, n_obj))
    pymoo_pop = PymooPop().new("X", X, "F", F)
    for i, ind in enumerate(population.individuals):
        pymoo_pop[i].F = np.array(ind.fitness, dtype=float)
        
    # run SMS-EMOA survival
    problem = NoConstraintProblem(n_obj)
    survivors = LeastHypervolumeContributionSurvival().do(
        problem,
        pymoo_pop,
        n_survive=N - 1
    )
    # extract survivor indices
    survivor_indices = [int(ind.X[0]) for ind in survivors]
    # map back to your population
    population.individuals = [population.individuals[i] for i in survivor_indices]