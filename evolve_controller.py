import numpy as np
import random
import copy
from scipy.stats import multivariate_normal
from utils import create_gif
import gymnasium as gym
import pandas as pd
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *

# ---- PARAMETERS ----
NUM_GENERATIONS = 200
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
            f'Generation {generation + 1}/{NUM_GENERATIONS}: Best Fitness: {fitness_scores[0]}, Average Fitness: {mean_fitness}'
        )
        results[generation] = [generation + 1, fitness_scores[0], mean_fitness]

    best_weights = pop[0]
    best_fitness = fitness_scores[0]

    return best_weights, best_fitness, results


def cma_es(pop_sz=20, nr_best=5):
    # TODO: Fix. Waiting for teacher to answear the email
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

    def flatten_weights(weights):
        """
        Flatten the weights into a single vector.
        """
        return np.concatenate([w.flatten() for w in weights])

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
    original_shape = [w.shape for w in pop[0]]  # Store the original shape of weights
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
            f'Generation {gen + 1}/{NUM_GENERATIONS}: Best Fitness: {fitness_scores[0]}, Average Fitness: {mean_fitness}'
        )
        results[gen] = [gen + 1, fitness_scores[0], mean_fitness]

    return pop


def run_genetic_algorithm():
    # Parameters
    population_size = 20
    mutation_strenght = 0.02
    mutation_rate = 0.05
    k_points = 2
    tournament_size = 5
    elite_size = 2

    # Example usage
    best_weights, best_fitness, results = genetic_algorithm(
        population_size,
        mutation_strenght,
        mutation_rate,
        k_points,
        tournament_size,
        elite_size,
    )

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

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', best_weights)

    print(f'Best weights vector found: {best_weights}')
    print(f'Best fitness score: {best_fitness}')

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif(
        robot_structure,
        filename=f'{path}\\ga.gif',
        scenario=SCENARIO,
        steps=STEPS,
        controller=brain,
    )

    for _ in range(10):
        visualize_policy(best_weights)

    return


def run_cma_es():
    # Parameters
    population_size = 20
    nr_best = 5

    # Example usage
    best_weights, best_fitness, results = cma_es(population_size, nr_best)

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

        f.write('Generation | Best Fitness | Avg Fitness\n')
        for it, best_fitness, avg_fitness in results:
            f.write(
                f'{int(it):<10} | {best_fitness:<12.6f} | {avg_fitness:.6f}\n'
            )

    # Save the best weights vector to a file
    np.save(f'{path}\\best_weights.npy', best_weights)

    print(f'Best weights vector found: {best_weights}')
    print(f'Best fitness score: {best_fitness}')

    set_weights(brain, best_weights)  # Load weights into the network
    create_gif(
        robot_structure,
        filename=f'{path}\\cma.gif',
        scenario=SCENARIO,
        steps=STEPS,
        controller=brain,
    )

    for _ in range(10):
        visualize_policy(best_weights)

    return


def main():
    start_time = time.time()
    # run_genetic_algorithm()
    cma_es()
    end_time = time.time()
    print("Elapsed time: " + str(end_time-start_time))
    return


if __name__ == '__main__':
    main()
