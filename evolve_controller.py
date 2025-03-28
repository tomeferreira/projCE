import numpy as np
import random
import copy
import gymnasium as gym
import pandas as pd
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *


NUM_GENERATIONS = 100  # Number of generations to evolve
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


def evolutionary_algorithm(
    population_size=20,
    mutation_strenght=0.02,
    mutation_rate=0.05,
    k_points=2,
    tournament_size=5,
    elite_size=2,
):
    def create_random_weights():
        return [np.random.randn(*param.shape) for param in brain.parameters()]

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

    results = np.zeros((NUM_GENERATIONS, 2))
    population = [create_random_weights() for _ in range(population_size)]
    population = sorted(population, key=evaluate_fitness, reverse=True)
    for generation in range(NUM_GENERATIONS):
        # Elitism: Retain the top N individuals
        new_population = copy.deepcopy(population[0:elite_size])
        # Generate the new population
        while len(new_population) < population_size:
            p1, p2 = tournament_selection(population, tournament_size)
            child = k_points_crossover(p1, p2, k_points)
            child = mutate(child, mutation_rate, mutation_strenght)
            new_population.append(child)

        # Evaluate the population
        population = sorted(new_population, key=evaluate_fitness, reverse=True)
        best_fitness = evaluate_fitness(population[0])
        print(
            f'Generation {generation + 1}/{NUM_GENERATIONS}, Best Fitness: {best_fitness}'
        )
        results[generation] = [generation + 1, best_fitness]

    best_weights = population[0]
    best_fitness = evaluate_fitness(population[0])

    return best_weights, best_fitness, results


def run_evolutionary_algorithm():
    # Parameters
    population_size = 20
    mutation_strenght = 0.02
    mutation_rate = 0.05
    k_points = 2
    tournament_size = 5
    elite_size = 2

    # Example usage
    best_weights, best_fitness, results = evolutionary_algorithm(
        population_size,
        mutation_strenght,
        mutation_rate,
        k_points,
        tournament_size,
        elite_size,
    )

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{STEPS}_G{NUM_GENERATIONS}'
    path = os.path.join('results', 'controller', 'ea', time_str)
    os.makedirs(path)

    # Save results to a text file
    with open('results.txt', 'w') as file:
        file.write(f'Population_Size: {population_size}\n')
        file.write(f'Mutation_Strength: {mutation_strenght}\n')
        file.write(f'Mutation_Rate: {mutation_rate}\n')
        file.write(f'K_Points: {k_points}\n')
        file.write(f'Tournament_Size: {tournament_size}\n')
        file.write(f'Elite_Size: {elite_size}\n\n')

        file.write("Iteration | Fitness\n")
        for it, fitness in results:
            file.write(f"{int(it):<9} | {fitness}\n")

    print(f'Best weights vector found: {best_weights}')
    print(f'Best fitness score: {best_fitness}')

    for _ in range(10):
        visualize_policy(best_weights)

    set_weights(brain, best_weights)  # Load weights into the network
    utils.create_gif(
        robot_structure,
        filename=f'{path}\ea.gif',
        scenario=SCENARIO,
        steps=STEPS,
        controller=brain,
    )

    return


def main():
    start_time = time.time()
    run_evolutionary_algorithm()
    end_time = time.time()
    print("Elapsed time: " + str(end_time-start_time))
    return


if __name__ == '__main__':
    main()
