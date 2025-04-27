import numpy as np
import random
import copy
from scipy.stats import multivariate_normal
from utils import create_gif_brain
import gymnasium as gym
import pandas as pd
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

# ---- PARAMETERS ----
NUM_GENERATIONS = 20
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])


connectivity = get_full_connectivity(robot_structure)
env = gym.make(SCENARIO, max_episode_steps=STEPS,
               body=robot_structure, connections=connectivity)
sim = env.sim
input_size = env.observation_space.shape[0]  # Observation size
output_size = env.action_space.shape[0]  # Action size

brain = NeuralController(input_size, output_size)


def evaluate_fitness(weights, view=False):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=robot_structure, connections=connectivity)
    sim = env
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    t_reward = 0
    for t in range(STEPS):
        # Update actuation before stepping
        state_tensor = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten()  # Get action
        if view:
            viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()
    return t_reward


def visualize_policy(weights):
    set_weights(brain, weights)  # Load weights into the network
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]  # Get initial state
    for t in range(STEPS):
        # Update actuation before stepping
        state_tensor = torch.tensor(
            state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action = brain(state_tensor).detach().numpy().flatten()  # Get action
        viewer.render('screen')
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
            break

    viewer.close()
    env.close()


def create_random_weights():
    return [np.random.randn(*param.shape) for param in brain.parameters()]


def flatten_weights(weights):
    """
    Flatten the weights into a single vector.
    """
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(weights, original_shape):
    """
    Unflatten the weights back to their original shape.
    """
    start = 0
    unflattened_weights = []
    for shape in original_shape:
        size = np.prod(shape)
        unflattened_weights.append(np.array(weights[start:start + size]).reshape(shape))
        start += size
    return unflattened_weights


def random_search(pop_sz=20):
    results = np.zeros((NUM_GENERATIONS, 3))

    for gen in range(NUM_GENERATIONS):
        # Generate random population
        pop = [create_random_weights() for _ in range(pop_sz)]

        # Evaluate fitness for the population
        fitness_scores = np.array([evaluate_fitness(ind) for ind in pop])
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        mean_fitness = np.mean(fitness_scores)

        # Logging
        print(
            f'Generation {gen + 1}/{NUM_GENERATIONS}: Best Fitness: {best_fitness:.4f}, Average Fitness: {mean_fitness:.4f}'
        )
        results[gen] = [gen + 1, best_fitness, mean_fitness]

    best_weights = pop[best_idx]

    return best_weights, best_fitness, results


def genetic_algorithm(
    pop_sz=20,
    mutation_strenght=0.02,
    mutation_rate=0.05,
    k_points=2,
    tournament_size=5,
    elite_size=2,
):
    def mutate(weights, mutation_rate=0.1, mutation_strength=0.02):
        mutated_weights = [w.copy() for w in weights]

        # Apply mutation to each array in the list
        for i, array in enumerate(mutated_weights):
            mutation_mask = np.random.rand(*array.shape) < mutation_rate
            noise = np.random.randn(*array.shape) * mutation_strength
            mutated_weights[i] += mutation_mask * noise

        return mutated_weights

    def k_points_crossover(parent1, parent2, k=2):
        # Perform k-points crossover between two parent weigths
        child = parent1.copy()
        points = np.random.choice(len(parent1), k, replace=False)
        points.sort()
        for i in range(k):
            if i % 2 == 0:
                # Swap the genes between the two parent weigths
                child[points[i]:points[i+1]] = parent2[points[i]:points[i+1]]
        return child

    def tournament_selection(population, k=5):
        # Select two parents using tournament selection
        chosen_idx = np.random.choice(len(population), k, replace=False)
        sorted_idx = sorted(chosen_idx)[:2]
        return population[sorted_idx[0]], population[sorted_idx[1]]

    results = np.zeros((NUM_GENERATIONS, 3))

    pop = [create_random_weights() for _ in range(pop_sz)]

    # Evaluate the initial population
    fitness_scores = np.array([evaluate_fitness(ind) for ind in pop])
    sorted_idx = np.argsort(-fitness_scores)
    pop = [pop[i] for i in sorted_idx]

    for generation in range(NUM_GENERATIONS):
        # Elitism: Retain the top N individuals
        new_pop = copy.deepcopy(pop[0:elite_size])
        # Generate the new population
        while len(new_pop) < pop_sz:
            p1, p2 = tournament_selection(pop, tournament_size)
            child = k_points_crossover(p1, p2, k_points)
            child = mutate(child, mutation_rate, mutation_strenght)
            new_pop.append(child)

        # Evaluate the population
        fitness_scores = np.array([evaluate_fitness(ind) for ind in new_pop])
        sorted_idx = np.argsort(-fitness_scores)
        fitness_scores = [fitness_scores[i] for i in sorted_idx]
        pop = [new_pop[i] for i in sorted_idx]
        mean_fitness = np.mean(fitness_scores)
        print(
            f'Generation {generation + 1}/{NUM_GENERATIONS}: Best Fitness: {fitness_scores[0]:.4f}, Average Fitness: {mean_fitness:.4f}'
        )
        results[generation] = [generation + 1, fitness_scores[0], mean_fitness]

    best_weights = pop[0]
    best_fitness = fitness_scores[0]

    return best_weights, best_fitness, results


def cma_es(pop_sz=20, nr_best=5):
    # TODO: Fix. Waiting for teacher to answear the email
    def estimate_cov_matrix(best_pop, miu_curr_gen=None):
        """
        Estimate the covariance matrix.
        """
        miu_next_gen = np.mean(best_pop, axis=0)
        cov_matrix = np.cov(best_pop, rowvar=False)
        return cov_matrix, miu_next_gen

    results = np.zeros((NUM_GENERATIONS, 3))

    # Initialize the population
    pop = [create_random_weights() for _ in range(pop_sz)]
    original_shape = [w.shape for w in pop[0]]
    # Flatten the population for easier manipulation
    pop = [flatten_weights(w) for w in pop]
    # Initialize the mean of the population
    miu = np.mean(pop, axis=0)
    dim = miu.shape[0]
    # Initialize the covariance matrix as an identity matrix
    cov_matrix = np.eye(dim)

    for gen in range(NUM_GENERATIONS):
        # Unflatten the population to evaluate fitness
        pop = [unflatten_weights(w, original_shape) for w in pop]
        # Evaluate fitness and sort population
        fitness_scores = np.array([evaluate_fitness(ind) for ind in pop])
        sorted_idx = np.argsort(-fitness_scores)
        pop = [pop[i] for i in sorted_idx]

        # Flatten the best individuals for covariance estimation
        best_pop = np.array(pop[:nr_best])
        best_pop = np.array([flatten_weights(w) for w in best_pop])

        # Calculate the covariance matrix and mean of the best individuals
        cov_matrix, miu = estimate_cov_matrix(best_pop)

        # Sample new population from multivariate normal distribution
        pop = multivariate_normal.rvs(mean=miu, cov=cov_matrix, size=pop_sz).tolist()

        # Logging
        mean_fitness = np.mean(fitness_scores)
        print(
            f'Generation {gen + 1}/{NUM_GENERATIONS}: Best Fitness: {fitness_scores[0]:.4f}, Average Fitness: {mean_fitness:.4f}'
        )
        results[gen] = [gen + 1, fitness_scores[0], mean_fitness]

    return pop


def differential_evolution(
    pop_sz=200,
    base_vector_type='random',
    nr_diff_vectors=1,
    scale_factor='random',
    scale_max=1,
    scale_min=0.5,
):
    def random_base_vector():
        """
        Get the indexes of a random selection of individuals for the differential vectors.
        The base (or target, as called in the slides) vector is the first one from the chosen ones.
        """
        nr_idx = nr_diff_vectors*2 + 1
        return np.random.choice(pop_sz, nr_idx, replace=False)

    def best_base_vector():
        """
        Get the indexes of the best individual and of a random selection of other individuals for the differential vectors.
        The base (or target, as called in the slides) vector is the first one from the chosen ones.
        """
        # Get the index of the best individual
        best_idx = np.argmax(fitnesses)
        nr_idx = nr_diff_vectors * 2

        # Create a list of all indices except the best index
        available_idxs = [i for i in range(pop_sz) if i != best_idx]

        # Randomly select indices from the available indices
        random_idxs = np.random.choice(available_idxs, nr_idx, replace=False).tolist()

        return [best_idx] + random_idxs

    def random_scale(current_gen):
        """
        Generate a random scale factor between 0.5 and 1.0.
        """
        # ? In the slides, it says that the scale factor for a random scale is between 0.5 and 1.0
        # ? but it can be between scale_min and scale_max
        # ? Sources say its mostly used between 0.5 and 1.0, but other values could be used
        # ? what to do
        return np.random.uniform(0.5, 1.0)

    def linear_scale(current_gen):
        """
        Generate a linear scale factor between scale_min and scale_max.
        """
        return scale_max - scale_min * ((NUM_GENERATIONS - current_gen) / NUM_GENERATIONS)

    def mutate(chosen_flat, scale):
        diff_vectors = []

        # Calculate the weighted difference vectors
        for i in range(1, len(chosen_flat), 2):
            diff_vector = scale * (chosen_flat[i] - chosen_flat[i + 1])
            diff_vectors.append(diff_vector)

        # Create the mutant vector by adding the weighted difference vectors to the base vector
        # The base vector is the first one in the chosen_flat list
        mutated_vector = chosen_flat[0] + np.sum(diff_vectors, axis=0)

        return mutated_vector

    def crossover(target, mutant):
        trial = target.copy()
        for i in range(len(target)):
            if np.random.rand() < 0.5:
                trial[i] = mutant[i]

        # Ensure at least one gene is changed
        if np.array_equal(trial, target):
            rand_idx = np.random.randint(len(target))
            trial[rand_idx] = mutant[rand_idx]

        return trial

    if base_vector_type == 'random':
        choose_base_vector = random_base_vector
    elif base_vector_type == 'best':
        choose_base_vector = best_base_vector
    else:
        raise ValueError("Invalid base vector type. Use 'random' or 'best'.")

    if nr_diff_vectors >= (pop_sz - 1) / 2:
        raise ValueError("Number of difference vectors must be less than half the population size.")

    if scale_factor == 'random':
        calculate_scale = random_scale
    elif scale_factor == 'linear':
        calculate_scale = linear_scale
    else:
        raise ValueError("Invalid scale factor type. Use 'random' or 'linear'.")

    results = np.zeros((NUM_GENERATIONS, 3))

    # 1. Generate a random population within bounds
    pop = [create_random_weights() for _ in range(pop_sz)]
    original_shape = [w.shape for w in pop[0]]

    # Evaluate the initial population
    fitnesses = np.zeros(pop_sz)
    for i, individual in enumerate(pop):
        r = evaluate_fitness(individual)
        fitnesses[i] = r

    for gen in range(NUM_GENERATIONS):
        new_pop = copy.deepcopy(pop)
        new_fitnesses = copy.deepcopy(fitnesses)

        for _ in range(pop_sz):
            scale = calculate_scale(gen)
            # Choose the base vector, and vectors used to calculate the differential vectors
            # The base (or target, as called in the slides) vector is the first one from the chosen ones
            chosen_idxs = choose_base_vector()

            # Flatten the weights of the chosen individuals for easier manipulation
            chosen_flat = [flatten_weights(pop[i]) for i in chosen_idxs]

            # 2. Create mutant (or variant, as called in the slides) vector by combining weighted differences of solutions
            mutant = mutate(chosen_flat, scale)

            # 3. Mix mutant vector with base vector to create trial solutions
            trial = crossover(chosen_flat[0], mutant)

            # 4. Choose the better solution between the base and trial vectors
            target_r = fitnesses[chosen_idxs[0]]
            # Unflatten the trial vector to evaluate fitness
            trial = unflatten_weights(trial, original_shape)
            trial_r = evaluate_fitness(trial)

            if trial_r > target_r:
                new_pop[chosen_idxs[0]] = trial
                new_fitnesses[chosen_idxs[0]] = trial_r

        pop = copy.deepcopy(new_pop)
        fitnesses = copy.deepcopy(new_fitnesses)

        # Logging
        best_fit_idx = np.argmax(fitnesses)
        mean_fitness = np.mean(fitnesses)
        print(
            f'Generation {gen + 1}/{NUM_GENERATIONS}: Best Fitness: {fitnesses[best_fit_idx]:.4f}, Average Fitness: {mean_fitness:.4f}'
        )
        results[gen] = [gen + 1, fitnesses[best_fit_idx], mean_fitness]

    return pop[best_fit_idx], fitnesses[best_fit_idx], results


def run_random_search():
    # Parameters
    population_size = 20

    print('Starting Random Search...')
    start_time = time.time()

    # Example usage
    best_weights, best_fitness, results = random_search(population_size)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{STEPS}_G{NUM_GENERATIONS}'
    path = os.path.join('results', 'controller', 'random', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open(f'{path}\\results.txt', 'w') as f:
        f.write(f'Generations: {NUM_GENERATIONS}\n')
        f.write(f'Steps: {STEPS}\n')
        f.write(f'Scenario: {SCENARIO}\n')
        f.write(f'Seed: {SEED}\n')
        f.write(f'Population_Size: {population_size}\n\n')
        f.write(f'Elapsed_Time: {elapsed_time} seconds\n\n')

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', best_weights)

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif_brain(
        robot_structure,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,
    )

    # visualize_policy(best_weights)

    return


def run_genetic_algorithm():
    # Parameters
    population_size = 20
    mutation_strenght = 0.02
    mutation_rate = 0.05
    k_points = 2
    tournament_size = 5
    elite_size = 2

    print('Starting Genetic Algorithm...')
    start_time = time.time()

    # Example usage
    best_weights, best_fitness, results = genetic_algorithm(
        population_size,
        mutation_strenght,
        mutation_rate,
        k_points,
        tournament_size,
        elite_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{STEPS}_G{NUM_GENERATIONS}'
    path = os.path.join('results', 'controller', 'ga', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open(f'{path}\\results.txt', 'w') as f:
        f.write(f'Generation: {NUM_GENERATIONS}\n')
        f.write(f'Steps: {STEPS}\n')
        f.write(f'Scenario: {SCENARIO}\n')
        f.write(f'Seed: {SEED}\n')
        f.write(f'Population_Size: {population_size}\n')
        f.write(f'Mutation_Strength: {mutation_strenght}\n')
        f.write(f'Mutation_Rate: {mutation_rate}\n')
        f.write(f'K_Points: {k_points}\n')
        f.write(f'Tournament_Size: {tournament_size}\n')
        f.write(f'Elite_Size: {elite_size}\n\n')
        f.write(f'Elapsed_Time: {elapsed_time} seconds\n\n')

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', best_weights)

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif_brain(
        robot_structure,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,
    )

    # visualize_policy(best_weights)

    return


def run_cma_es():
    # Parameters
    population_size = 20
    nr_best = 5

    print('Starting CMA-ES...')
    start_time = time.time()

    # Example usage
    best_weights, best_fitness, results = cma_es(population_size, nr_best)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{STEPS}_G{NUM_GENERATIONS}'
    path = os.path.join('results', 'controller', 'cma', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open(f'{path}\\results.txt', 'w') as f:
        f.write(f'Generation: {NUM_GENERATIONS}\n')
        f.write(f'Steps: {STEPS}\n')
        f.write(f'Scenario: {SCENARIO}\n')
        f.write(f'Seed: {SEED}\n')
        f.write(f'Population_Size: {population_size}\n')
        f.write(f'Nr_Best: {nr_best}\n\n')
        f.write(f'Elapsed_Time: {elapsed_time} seconds\n\n')

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', best_weights)

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif_brain(
        robot_structure,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,
    )

    # visualize_policy(best_weights)

    return


def run_differential_evolution():
    # Parameters
    population_size = 20
    base_vector_type = 'best'  # 'random' or 'best'
    nr_diff_vectors = 3
    scale_factor = 'linear'  # 'random' or 'linear'
    scale_max = 1
    scale_min = 0.5

    print('Starting Differential Evolution...')
    start_time = time.time()

    # Example usage
    best_weights, best_fitness, results = differential_evolution(
        population_size,
        base_vector_type,
        nr_diff_vectors,
        scale_factor,
        scale_max,
        scale_min,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{STEPS}_G{NUM_GENERATIONS}'
    path = os.path.join('results', 'controller', 'de', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open(f'{path}\\results.txt', 'w') as f:
        f.write(f'Generations: {NUM_GENERATIONS}\n')
        f.write(f'Steps: {STEPS}\n')
        f.write(f'Scenario: {SCENARIO}\n')
        f.write(f'Seed: {SEED}\n')
        f.write(f'Population_Size: {population_size}\n')
        f.write(f'Base_Vector_Type: {base_vector_type}\n')
        f.write(f'Nr_Diff_Vectors: {nr_diff_vectors}\n')
        f.write(f'Scale_Factor: {scale_factor}\n')
        f.write(f'Scale_Max: {scale_max}\n')
        f.write(f'Scale_Min: {scale_min}\n\n')
        f.write(f'Elapsed_Time: {elapsed_time} seconds\n\n')

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', np.array(best_weights, dtype=object))

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif_brain(
        robot_structure,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,
    )

    # visualize_policy(best_weights)

    return


def main():
    run_random_search()
    # run_genetic_algorithm()
    # run_cma_es()
    # run_differential_evolution()
    return


if __name__ == '__main__':
    main()
