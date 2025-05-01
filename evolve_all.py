import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, sample_robot, is_connected
from neural_controller import *
import utils
import os
import time
import torch

# ---- PARAMETERS ----
NUM_GENERATIONS = 100
STEPS = 500
SCENARIO = 'Walker-v0'
SEEDS = [68167, 91803, 39233, 58400, 97664]
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set")
    return


def calculate_param_shapes(structure):
    """Calculate the shapes of the parameters of the neural controller for a given structure."""
    connectivity = get_full_connectivity(structure)
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=structure, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    return [param.shape for param in brain.parameters()]


def create_random_robot():
    """Generate a valid random robot structure."""
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
                 random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


def create_random_weights(structures):
    """Generate random weights for the neural controller."""
    # ! NOT WORKING
    param_shape = calculate_param_shapes()
    return [np.random.randn(*shape) for shape in param_shape]


def evaluate_fitness(robot_structure, weights, view=False):
    """Evaluate the fitness of a robot with a given structure and controller weights."""
    try:
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS,
                       body=robot_structure, connections=connectivity)
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        brain = NeuralController(input_size, output_size)
        set_weights(brain, weights)

        state = env.reset()[0]  # Get initial state
        t_reward = 0
        for t in range(STEPS):
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
    except (ValueError, IndexError) as e:
        return 0.0


def coevolution(pop_sz, nr_gens, mutation_rate_structure, mutation_rate_weights, elitism_sz, tournament_sz):
    """Run the coevolutionary algorithm."""
    def mutate_structure(robot_structure, mutation_rate):
        """Mutate the robot structure."""
        mutated = copy.deepcopy(robot_structure)
        for i in range(mutated.shape[0]):
            for j in range(mutated.shape[1]):
                if random.random() < mutation_rate:
                    mutated[i][j] = random.choice(VOXEL_TYPES)
        while not is_connected(mutated):
            mutated = copy.deepcopy(robot_structure)
            for i in range(mutated.shape[0]):
                for j in range(mutated.shape[1]):
                    if random.random() < mutation_rate:
                        mutated[i][j] = random.choice(VOXEL_TYPES)
        return mutated

    def mutate_weights(weights, mutation_rate):
        """Mutate the controller weights."""
        mutated_weights = [w.copy() for w in weights]
        for i, array in enumerate(mutated_weights):
            mutation_mask = np.random.rand(*array.shape) < mutation_rate
            noise = np.random.randn(*array.shape) * 0.02  # Mutation strength
            mutated_weights[i] += mutation_mask * noise
        return mutated_weights

    def crossover_structure(parent1, parent2):
        """Perform crossover on robot structures."""
        child = copy.deepcopy(parent1)
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                if random.random() < 0.5:
                    child[i][j] = parent2[i][j]
        while not is_connected(child):
            child = crossover_structure(parent1, parent2)
        return child

    def crossover_weights(parent1, parent2):
        """Perform crossover on controller weights."""
        child = [w1.copy() for w1 in parent1]
        for i in range(len(child)):
            mask = np.random.rand(*child[i].shape) < 0.5
            child[i][mask] = parent2[i][mask]
        return child

    def tournament_selection(pop, fitness_scores, k):
        """Select two parents using tournament selection."""
        indices = np.random.choice(len(pop), k, replace=False)
        sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
        return pop[sorted_indices[0]], pop[sorted_indices[1]]

    pop = [(create_random_robot(), create_random_weights()) for _ in range(pop_sz)]
    fitness_scores = [evaluate_fitness(robot, weights) for robot, weights in pop]

    for generation in range(nr_gens):
        new_population = []
        sorted_indices = np.argsort(fitness_scores)[::-1]
        pop = [pop[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Elitism
        new_population.extend(pop[:elitism_sz])

        # Generate new population
        while len(new_population) < pop_sz:
            parent1, parent2 = tournament_selection(pop, fitness_scores, tournament_sz)
            child_structure = crossover_structure(parent1[0], parent2[0])
            child_weights = crossover_weights(parent1[1], parent2[1])
            child_structure = mutate_structure(child_structure, mutation_rate_structure)
            child_weights = mutate_weights(child_weights, mutation_rate_weights)
            new_population.append((child_structure, child_weights))

        pop = new_population
        fitness_scores = [evaluate_fitness(robot, weights) for robot, weights in pop]

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        print(
            f"Generation {generation + 1}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

    best_robot, best_weights = pop[np.argmax(fitness_scores)]
    return best_robot, best_weights


def run_coevolution(seed):
    # Parameters
    pop_size = 50
    num_generations = 100
    mutation_rate_structure = 0.1
    mutation_rate_weights = 0.1
    elitism_size = 2
    tournament_size = 5

    print('Starting Coevolution...')
    start_time = time.time()

    best_robot, best_weights = coevolution(
        pop_size,
        num_generations,
        mutation_rate_structure,
        mutation_rate_weights,
        elitism_size,
        tournament_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{seed}'
    path = os.path.join('results', 'all', 'coevolution', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open(f'{path}\\results.txt', 'w') as f:
        f.write(f'Generations: {num_generations}\n')
        f.write(f'Population Size: {pop_size}\n')
        f.write(f'Mutation Rate (Structure): {mutation_rate_structure}\n')
        f.write(f'Mutation Rate (Weights): {mutation_rate_weights}\n')
        f.write(f'Elitism Size: {elitism_size}\n')
        f.write(f'Tournament Size: {tournament_size}\n')
        f.write(f'Seed: {seed}\n\n')
        f.write(f'Elapsed Time: {elapsed_time} seconds\n')

    # Save the best robot structure and weights
    np.save(f'{path}\\best_robot.npy', best_robot)
    np.save(f'{path}\\best_weights.npy', best_weights)

    return


def main():
    for seed in SEEDS:
        set_seed(seed)
        run_coevolution(seed)
    return


if __name__ == '__main__':
    main()
