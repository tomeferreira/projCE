from concurrent.futures import ProcessPoolExecutor
from cmaes import CMA
from functools import partial
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
from utils import create_gif_brain
import json
import pandas as pd

# ---- PARAMETERS ----
NUM_CORES = 10
ELITISM_SIZE = 2
STEPS = 500
SCENARIO = 'GapJumper-v0'
SEEDS = [68167, 91803, 39233, 58400, 97664]
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)
MIN_GRID_SIZE = (5, 5)
MAX_GRID_SIZE = (5, 5)
MAX_INPUT = 72  # 78
MAX_OUTPUT = 10  # 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set")
    return


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


def get_brain_dimensions(weights):
    input_size = weights[0].shape[1]
    output_size = weights[-1].shape[0]
    return input_size, output_size


def calculate_param_shapes(structure):
    """Calculate the shapes of the parameters of the neural controller for a given structure."""
    connectivity = get_full_connectivity(structure)
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=structure, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    # brain = NeuralController(MAX_INPUT, MAX_OUTPUT)
    return [param.shape for param in brain.parameters()]


def create_random_robot():
    """Generate a valid random robot structure."""
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
                 random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


def create_random_weights(param_shape):
    """Generate random weights for the neural controller."""
    # ! NOT WORKING
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


def coevolution_v6(pop_struct_sz, pop_brain_sz, nr_gens_struct, nr_gens_brain, mutation_rate_structure, mutation_rate_weights, mutation_strength_weights, elitism_sz, tournament_sz):
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

    def crossover_structure(parent1, parent2):
        # Flatten parents to 1D arrays
        parent1_flat = copy.deepcopy(parent1).flatten()
        parent2_flat = copy.deepcopy(parent2).flatten()

        mask = np.random.randint(0, 2, size=25)
        child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

        i = 1
        while not is_connected(child_flat.reshape((5, 5))):
            # print("Wasn't Connected Crossover")
            if i > 5:
                child_flat = create_random_robot()
                break
            mask = np.random.randint(0, 2, size=25)
            child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
            i += 1

        # Reshape back to (5,5)
        child = child_flat.reshape((5, 5))

        return child

    def tournament_selection(pop, fitness_scores, k):
        """Select two parents using tournament selection."""
        indices = np.random.choice(len(pop), k, replace=False)
        sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
        return pop[sorted_indices[0]], pop[sorted_indices[1]]

    def genetic_algorithm(
        struct,
        gens=10,
        pop_sz=20,
        mutation_strenght=0.02,
        mutation_rate=0.05,
        k_points=2,
        tournament_size=5,
        elite_size=2,
    ):
        def mutate(weights, mutation_rate=0.1, mutation_strength=0.02):
            mutated_weights = [copy.deepcopy(w) for w in weights]

            # Apply mutation to each array in the list
            for i, array in enumerate(mutated_weights):
                mutation_mask = np.random.rand(*array.shape) < mutation_rate
                noise = np.random.randn(*array.shape) * mutation_strength
                mutated_weights[i] += mutation_mask * noise

            return mutated_weights

        def k_points_crossover(parent1, parent2, k=2):
            # Perform k-points crossover between two parent weigths
            child = copy.deepcopy(parent1)
            points = np.random.choice(len(parent1), k, replace=False)
            points.sort()
            for i in range(k):
                if i % 2 == 0:
                    # Swap the genes between the two parent weigths
                    child[points[i]:points[i+1]] = parent2[points[i]:points[i+1]]
            return child

        param_shape = calculate_param_shapes(struct)
        pop = [create_random_weights(param_shape) for _ in range(pop_brain_sz)]
        pairings = [(struct, pair) for pair in pop]

        # Evaluate the initial population
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            fitness_scores = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

        sorted_idx = np.argsort(-fitness_scores)
        fitness_scores = [fitness_scores[i] for i in sorted_idx]
        pop = [pop[i] for i in sorted_idx]
        pairings = [pairings[i] for i in sorted_idx]

        best_fitness = fitness[0]
        avg_fitness = np.mean(fitness)
        print(
            f"Generation 0/{gens}: - Best Fitness = {best_fitness:.4f}, Average Fitness = {avg_fitness:.4f}")

        for generation in range(gens):
            elite_individuals = pop[:elite_size]
            elite_fitness_scores = copy.deepcopy(fitness_scores[:elite_size])

            new_pop = copy.deepcopy(elite_individuals)
            # Generate the new population
            while len(new_pop) < pop_sz:
                p1, p2 = tournament_selection(pop, fitness_scores, tournament_size)
                child = k_points_crossover(p1, p2, k_points)
                child = mutate(child, mutation_rate, mutation_strenght)
                new_pop.append(child)

            non_elite_individuals = new_pop[elite_size:]
            pairings = [(struct, individual) for individual in non_elite_individuals]

            # Evaluate the population
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                non_elite_fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

            fitness_scores = np.concatenate((elite_fitness_scores, non_elite_fitness))

            sorted_idx = np.argsort(-fitness_scores)
            fitness_scores = [fitness_scores[i] for i in sorted_idx]
            pop = [new_pop[i] for i in sorted_idx]
            pairings = [pairings[i] for i in sorted_idx]
            mean_fitness = np.mean(fitness_scores)
            print(
                f'Generation {generation + 1}/{gens}: Best Fitness: {fitness_scores[0]:.4f}, Average Fitness: {mean_fitness:.4f}')

        best_weights = pop[0]
        best_fitness = fitness_scores[0]

        return best_weights, best_fitness

    results = np.zeros((nr_gens_struct+1, 3))

    pop_struct = [create_random_robot() for _ in range(pop_struct_sz)]
    pairings = []
    fitness = []

    for i, struct in enumerate(pop_struct):
        print(f"\n\n Robot {str(i)}\n\n")
        best_weights, best_fitness = genetic_algorithm(
            struct,
            nr_gens_brain,
            pop_brain_sz,
            mutation_strength_weights,
            mutation_rate_weights,
            2,
            tournament_sz,
            elitism_sz,
        )
        pairings.append((struct, best_weights))
        fitness.append(best_fitness)
    sorted_idx = np.argsort(-np.array(fitness))
    pop_struct = [pop_struct[i] for i in sorted_idx]
    pairings = [pairings[i] for i in sorted_idx]

    print(fitness)
    best_fitness = fitness[0]
    avg_fitness = np.mean(fitness)
    print(
        f"Generation 0: - Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

    results[0] = [0, best_fitness, np.mean(fitness)]

    for generation in range(nr_gens_struct):
        # Elitism
        new_population_struct = list(copy.deepcopy(pop_struct[0:ELITISM_SIZE]))

        # Generate new population struct
        while len(new_population_struct) < pop_struct_sz:
            parent1, parent2 = tournament_selection(pop_struct, fitness, tournament_sz)
            child_structure = crossover_structure(parent1, parent2)
            child_structure = mutate_structure(child_structure, mutation_rate_structure)
            new_population_struct.append(child_structure)
        pop_struct = new_population_struct

        pairings = pairings[0:ELITISM_SIZE]
        fitness = fitness[0:ELITISM_SIZE]

        for i, struct in enumerate(pop_struct[ELITISM_SIZE:]):
            print(f"\n\n Robot {str(i)}\n\n")
            best_weights, best_fitness = genetic_algorithm(
                struct,
                nr_gens_brain,
                pop_brain_sz,
                mutation_strength_weights,
                mutation_rate_weights,
                2,
                tournament_sz,
                elitism_sz,
            )
            pairings.append((struct, best_weights))
            fitness.append(best_fitness)
        sorted_idx = np.argsort(-np.array(fitness))
        fitness = [fitness[i] for i in sorted_idx]
        pop_struct = [pop_struct[i] for i in sorted_idx]
        pairings = [pairings[i] for i in sorted_idx]

        print(fitness)
        best_fitness = fitness[0]
        avg_fitness = np.mean(fitness)
        print(
            f"Generation {generation + 1}: - Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

        results[generation+1] = [generation+1, fitness[0], np.mean(fitness)]

    best_robot = pairings[0][0]
    best_weights = pairings[0][1]
    best_fitness = fitness[0]

    results = pd.DataFrame(results, columns=['Iteration', 'Best Reward', 'Mean Reward'])
    results['Iteration'] = results['Iteration'].astype('int32')

    return best_robot, best_weights, best_fitness, results


def coevolution_cma(pop_struct_sz, pop_brain_sz, nr_gens_struct, nr_gens_brain, mutation_rate_structure, tournament_sz):
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

    def crossover_structure(parent1, parent2):
        # Flatten parents to 1D arrays
        parent1_flat = copy.deepcopy(parent1).flatten()
        parent2_flat = copy.deepcopy(parent2).flatten()

        mask = np.random.randint(0, 2, size=25)
        child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

        i = 1
        while not is_connected(child_flat.reshape((5, 5))):
            # print("Wasn't Connected Crossover")
            if i > 5:
                child_flat = create_random_robot()
                break
            mask = np.random.randint(0, 2, size=25)
            child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
            i += 1

        # Reshape back to (5,5)
        child = child_flat.reshape((5, 5))

        return child

    def tournament_selection(pop, fitness_scores, k):
        """Select two parents using tournament selection."""
        indices = np.random.choice(len(pop), k, replace=False)
        sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
        return pop[sorted_indices[0]], pop[sorted_indices[1]]

    def controller_cma(struct, gens=10):
        param_shape = calculate_param_shapes(struct)
        dim = sum(np.prod(shape) for shape in param_shape)  # Flattened dimension
        optimizer = CMA(mean=np.zeros(dim), sigma=1.5, population_size=pop_brain_sz)

        # To store generation, best fitness, and average fitness
        results = np.zeros((gens, 3))

        best_solution = None
        best_fitness = -np.inf  # Initialize with a very low fitness value

        for generation in range(gens):
            solutions = []
            fitness_scores = []

            # Generate solutions
            flat_weights_list = [optimizer.ask() for _ in range(optimizer.population_size)]

            pairings = [(struct, unflatten_weights(w, param_shape)) for w in flat_weights_list]

            # Evaluate solutions
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                fitness_results = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

            # Process the results
            for flat_weights, fitness in zip(flat_weights_list, fitness_results):
                # Negate fitness for minimization
                solutions.append((flat_weights, -fitness))
                fitness_scores.append(fitness)

                # Update the best solution if a better one is found
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = flat_weights

            # Update the optimizer with the evaluated solutions
            optimizer.tell(solutions)

            # Logging
            avg_fitness = np.mean(fitness_scores)
            print(
                f"Generation {generation + 1}/{gens}: Best Fitness = {best_fitness:.4f}, Average Fitness = {avg_fitness:.4f}")
            results[generation] = [generation + 1, best_fitness, avg_fitness]

        best_weights = unflatten_weights(best_solution, param_shape)
        return best_weights, best_fitness, results

    results = np.zeros((nr_gens_struct+1, 3))

    pop_struct = [create_random_robot() for _ in range(pop_struct_sz)]
    pairings = []
    fitness = []

    for i, struct in enumerate(pop_struct):
        print(f"\n\n Robot {str(i)}\n\n")
        best_weights, best_fitness, results = controller_cma(struct, nr_gens_brain)
        pairings.append((struct, best_weights))
        fitness.append(best_fitness)
    sorted_idx = np.argsort(-np.array(fitness))
    fitness = [fitness[i] for i in sorted_idx]
    pop_struct = [pop_struct[i] for i in sorted_idx]
    pairings = [pairings[i] for i in sorted_idx]

    best_fitness_struct = fitness[0]
    avg_fitness_struct = np.mean(fitness)
    print(
        f"Generation 0: - Best Fitness = {best_fitness_struct}, Average Fitness = {avg_fitness_struct}")

    results[0] = [0, fitness[0], np.mean(fitness)]

    for generation in range(nr_gens_struct):
        # Elitism
        new_population_struct = list(copy.deepcopy(pop_struct[0:ELITISM_SIZE]))

        # Generate new population struct
        while len(new_population_struct) < pop_struct_sz:
            parent1, parent2 = tournament_selection(pop_struct, fitness, tournament_sz)
            child_structure = crossover_structure(parent1, parent2)
            child_structure = mutate_structure(child_structure, mutation_rate_structure)
            new_population_struct.append(child_structure)
        pop_struct = new_population_struct
        
        pairings = pairings[0:ELITISM_SIZE]
        fitness = fitness[0:ELITISM_SIZE]

        for i, struct in enumerate(pop_struct[ELITISM_SIZE:]):
            print(f"\n\n Robot {str(i)}\n\n")
            best_weights, best_fitness, results = controller_cma(struct, nr_gens_brain)
            pairings.append((struct, best_weights))
            fitness.append(best_fitness)
        sorted_idx = np.argsort(-np.array(fitness))
        fitness = [fitness[i] for i in sorted_idx]
        pop_struct = [pop_struct[i] for i in sorted_idx]
        pairings = [pairings[i] for i in sorted_idx]

        best_fitness_struct = fitness[0]
        avg_fitness_struct = np.mean(fitness)
        print(
            f"Generation {generation + 1}: - Best Fitness = {best_fitness_struct}, Average Fitness = {avg_fitness_struct}")

        results[generation+1] = [generation+1, fitness[0], np.mean(fitness)]

    best_robot = pairings[0][0]
    best_weights = pairings[0][1]
    best_fitness = fitness[0]

    results = pd.DataFrame(results, columns=['Iteration', 'Best Reward', 'Mean Reward'])
    results['Iteration'] = results['Iteration'].astype('int32')

    return best_robot, best_weights, best_fitness, results

# def coevolution_v5(pop_struct_sz, pop_brain_sz, nr_gens, mutation_rate_structure, mutation_rate_weights, elitism_sz, tournament_sz):
#     """Run the coevolutionary algorithm."""
#     def mutate_structure(robot_structure, mutation_rate):
#         """Mutate the robot structure."""
#         mutated = copy.deepcopy(robot_structure)
#         for i in range(mutated.shape[0]):
#             for j in range(mutated.shape[1]):
#                 if random.random() < mutation_rate:
#                     mutated[i][j] = random.choice(VOXEL_TYPES)
#         while not is_connected(mutated):
#             mutated = copy.deepcopy(robot_structure)
#             for i in range(mutated.shape[0]):
#                 for j in range(mutated.shape[1]):
#                     if random.random() < mutation_rate:
#                         mutated[i][j] = random.choice(VOXEL_TYPES)
#         return mutated

#     def mutate_weights(weights, mutation_rate):
#         """Mutate the controller weights."""
#         mutated_weights = [copy.deepcopy(w) for w in weights]
#         for i, array in enumerate(mutated_weights):
#             mutation_mask = np.random.rand(*array.shape) < mutation_rate
#             noise = np.random.randn(*array.shape) * 0.02  # Mutation strength
#             mutated_weights[i] += mutation_mask * noise
#         return mutated_weights

#     def crossover_structure(parent1,parent2):
#         # Flatten parents to 1D arrays
#         parent1_flat = copy.deepcopy(parent1).flatten()
#         parent2_flat = copy.deepcopy(parent2).flatten()

#         mask = np.random.randint(0, 2, size=25)
#         child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

#         i = 1
#         while not is_connected(child_flat.reshape((5,5))):
#             #print("Wasn't Connected Crossover")
#             if i>5:
#                 child_flat = create_random_robot()
#                 break
#             mask = np.random.randint(0, 2, size=25)
#             child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
#             i+=1

#         # Reshape back to (5,5)
#         child = child_flat.reshape((5,5))

#         return child

#     def crossover_weights(parent1, parent2):
#         """Perform crossover on controller weights."""
#         child = [copy.deepcopy(w1) for w1 in parent1]
#         for i in range(len(child)):
#             mask = np.random.rand(*child[i].shape) < 0.5
#             child[i][mask] = parent2[i][mask]
#         return child
#         # return parent1

#     def tournament_selection(pop, fitness_scores, k):
#         """Select two parents using tournament selection."""
#         indices = np.random.choice(len(pop), k, replace=False)
#         sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
#         return pop[sorted_indices[0]], pop[sorted_indices[1]]

#     def random_paring(pop_s,pop_b):
#         fitness_w1 = []
#         pairings = []

#         # Evaluation of pop_w1
#         for w1 in pop_s:
#             indices = np.random.choice(len(pop_b),size=3,replace=False) # choose randomly the partner
#             partners = [pop_b[i] for i in indices]
#             for partner in partners:
#                 pairings.append((w1,partner))
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w1 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         fitness_w1 = fitness_w1.reshape(len(pop_s), 3).mean(axis=1)

#         fitness_w2 = []
#         pairings = []

#         # Evaluation of pop_w2
#         for w2 in pop_b:
#             indices = np.random.choice(len(pop_s),size=10,replace=False) # choose randomly the partner
#             partners = [pop_s[i] for i in indices]
#             placed = 0
#             not_used = []
#             for partner in partners:
#                 if placed>=3:
#                     break
#                 connectivity = get_full_connectivity(partner)
#                 env = gym.make(SCENARIO, max_episode_steps=STEPS,
#                             body=partner, connections=connectivity)
#                 input_size = env.observation_space.shape[0]
#                 output_size = env.action_space.shape[0]
#                 if MAX_INPUT==input_size and MAX_OUTPUT==output_size:
#                     pairings.append((partner,w2))
#                     placed+=1
#                 else:
#                     not_used.append(partner)
#             i=0
#             while placed<3:
#                 pairings.append((not_used[i],w2))
#                 placed+=1
#                 i+=1

#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w2 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         fitness_w2 = fitness_w2.reshape(len(pop_b), 3).mean(axis=1)

#         return fitness_w1,fitness_w2

#     results = np.zeros((nr_gens+1,5))

#     pop_struct = [create_random_robot() for _ in range(pop_struct_sz)]
#     pop_brain = [create_random_weights(create_random_robot()) for _ in range(pop_brain_sz)]

#     fitness_scores_struct, fitness_scores_brain = random_paring(pop_struct,pop_brain)

#     for generation in range(nr_gens):
#         new_population_struct = []
#         new_population_brain = []
#         sorted_indices_struct = np.argsort(fitness_scores_struct)[::-1]
#         sorted_indices_brain = np.argsort(fitness_scores_brain)[::-1]
#         pop_struct = [pop_struct[i] for i in sorted_indices_struct]
#         fitness_scores_struct = [fitness_scores_struct[i] for i in sorted_indices_struct]
#         pop_brain = [pop_brain[i] for i in sorted_indices_brain]
#         fitness_scores_brain = [fitness_scores_brain[i] for i in sorted_indices_brain]

#         # Elitism
#         new_population_struct = list(copy.deepcopy(pop_struct[0:ELITISM_SIZE]))
#         new_population_brain = list(copy.deepcopy(pop_brain[0:ELITISM_SIZE]))

#         # Generate new population struct
#         while len(new_population_struct) < pop_struct_sz:
#             parent1, parent2 = tournament_selection(pop_struct, fitness_scores_struct, tournament_sz)
#             child_structure = crossover_structure(parent1, parent2)
#             child_structure = mutate_structure(child_structure, mutation_rate_structure)
#             new_population_struct.append(child_structure)
#         pop_struct = new_population_struct

#         # Generate new population brain
#         while len(new_population_brain) < pop_brain_sz:
#             parent1, parent2 = tournament_selection(pop_brain, fitness_scores_brain, tournament_sz)
#             child_weights = crossover_weights(parent1, parent2)
#             child_weights = mutate_weights(child_weights, mutation_rate_weights)
#             new_population_brain.append(child_weights)
#         pop_brain = new_population_brain

#         fitness_scores_struct,fitness_scores_brain = random_paring(pop_struct,pop_brain)

#         best_fitness_struct = max(fitness_scores_struct)
#         avg_fitness_struct = np.mean(fitness_scores_struct)
#         print(
#             f"Generation {generation + 1}: Structure - Best Fitness = {best_fitness_struct}, Average Fitness = {avg_fitness_struct}")
#         best_fitness_brain = max(fitness_scores_brain)
#         avg_fitness_brain = np.mean(fitness_scores_brain)
#         print(
#             f"Generation {generation + 1}: Brain - Best Fitness = {best_fitness_brain}, Average Fitness = {avg_fitness_brain}")

#         results[generation] = [generation+1,fitness_scores_struct[0],np.mean(fitness_scores_struct),fitness_scores_brain[0],np.mean(fitness_scores_brain)]

#     final_pairings = []
#     for structure in pop_struct:
#         for brain in pop_brain:
#             final_pairings.append((structure,brain))
#     with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#         fitness = np.array(list(executor.map(evaluate_fitness, *zip(*final_pairings))))

#     best_robot = final_pairings[np.argmax(fitness)][0]
#     best_weights = final_pairings[np.argmax(fitness)][1]
#     best_fitness = max(fitness)

#     results[nr_gens] = [nr_gens+1,best_fitness,np.mean(fitness),best_fitness,np.mean(fitness)]

#     results = pd.DataFrame(results,columns=['Iteration','Best Reward Struct','Mean Reward Struct','Best Reward Brain','Mean Reward Brain'])
#     results['Iteration'] = results['Iteration'].astype('int32')

#     return best_robot, best_weights, best_fitness, results

# def coevolution_v4(pop_struct_sz, pop_brain_sz, nr_gens, mutation_rate_structure, mutation_rate_weights, elitism_sz, tournament_sz):
#     """Run the coevolutionary algorithm."""
#     def mutate_structure(robot_structure, mutation_rate):
#         """Mutate the robot structure."""
#         mutated = copy.deepcopy(robot_structure)
#         for i in range(mutated.shape[0]):
#             for j in range(mutated.shape[1]):
#                 if random.random() < mutation_rate:
#                     mutated[i][j] = random.choice(VOXEL_TYPES)
#         while not is_connected(mutated):
#             mutated = copy.deepcopy(robot_structure)
#             for i in range(mutated.shape[0]):
#                 for j in range(mutated.shape[1]):
#                     if random.random() < mutation_rate:
#                         mutated[i][j] = random.choice(VOXEL_TYPES)
#         return mutated

#     def mutate_weights(weights, mutation_rate):
#         """Mutate the controller weights."""
#         mutated_weights = [copy.deepcopy(w) for w in weights]
#         for i, array in enumerate(mutated_weights):
#             mutation_mask = np.random.rand(*array.shape) < mutation_rate
#             noise = np.random.randn(*array.shape) * 0.02  # Mutation strength
#             mutated_weights[i] += mutation_mask * noise
#         return mutated_weights

#     def crossover_structure(parent1,parent2):
#         # Flatten parents to 1D arrays
#         parent1_flat = copy.deepcopy(parent1).flatten()
#         parent2_flat = copy.deepcopy(parent2).flatten()

#         mask = np.random.randint(0, 2, size=25)
#         child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

#         i = 1
#         while not is_connected(child_flat.reshape((5,5))):
#             #print("Wasn't Connected Crossover")
#             if i>5:
#                 child_flat = create_random_robot()
#                 break
#             mask = np.random.randint(0, 2, size=25)
#             child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
#             i+=1

#         # Reshape back to (5,5)
#         child = child_flat.reshape((5,5))

#         return child

#     def crossover_weights(parent1, parent2):
#         """Perform crossover on controller weights."""
#         child = [copy.deepcopy(w1) for w1 in parent1]
#         for i in range(len(child)):
#             mask = np.random.rand(*child[i].shape) < 0.5
#             child[i][mask] = parent2[i][mask]
#         return child
#         # return parent1

#     def tournament_selection(pop, fitness_scores, k):
#         """Select two parents using tournament selection."""
#         indices = np.random.choice(len(pop), k, replace=False)
#         sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
#         return pop[sorted_indices[0]], pop[sorted_indices[1]]

#     def random_paring(pop_s,pop_b):
#         fitness_w1 = []
#         pairings = []

#         # Evaluation of pop_w1
#         for w1 in pop_s:
#             partner = random.choice(pop_b) # choose randomly the partner
#             pairings.append((w1,partner))
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w1 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         fitness_w2 = []
#         pairings = []

#         # Evaluation of pop_w2
#         for w2 in pop_b:
#             partner = random.choice(pop_s) # choose randomly the partner
#             pairings.append((partner,w2))
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w2 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         return fitness_w1,fitness_w2

#     def best_so_far_paring_individual(pop_s,pop_b,eval_pop_1=True):
#         pairings = []

#         if eval_pop_1:
#             for w1 in pop_s:
#                 partners = pop_b[0:3]
#                 for partner in partners:
#                     pairings.append((w1,partner))
#             with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#                 fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))
#             fitness = fitness.reshape(len(pop_s), 3).mean(axis=1)
#         else:
#             for w2 in pop_b:
#                 partners = pop_s[0:3]
#                 for partner in partners:
#                     pairings.append((partner,w2))
#             with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#                 fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))
#             fitness = fitness.reshape(len(pop_b), 3).mean(axis=1)

#         return fitness

#     results = np.zeros((nr_gens+1,5))

#     pop_struct = [create_random_robot() for _ in range(pop_struct_sz)]
#     pop_brain = [create_random_weights(create_random_robot()) for _ in range(pop_brain_sz)]

#     fitness_scores_struct, fitness_scores_brain = random_paring(pop_struct,pop_brain)

#     for generation in range(nr_gens):
#         new_population_struct = []
#         new_population_brain = []
#         sorted_indices_struct = np.argsort(fitness_scores_struct)[::-1]
#         sorted_indices_brain = np.argsort(fitness_scores_brain)[::-1]
#         pop_struct = [pop_struct[i] for i in sorted_indices_struct]
#         fitness_scores_struct = [fitness_scores_struct[i] for i in sorted_indices_struct]
#         pop_brain = [pop_brain[i] for i in sorted_indices_brain]
#         fitness_scores_brain = [fitness_scores_brain[i] for i in sorted_indices_brain]

#         # Elitism
#         new_population_struct = list(copy.deepcopy(pop_struct[0:ELITISM_SIZE]))
#         new_population_brain = list(copy.deepcopy(pop_brain[0:ELITISM_SIZE]))

#         # Generate new population struct
#         while len(new_population_struct) < pop_struct_sz:
#             parent1, parent2 = tournament_selection(pop_struct, fitness_scores_struct, tournament_sz)
#             child_structure = crossover_structure(parent1, parent2)
#             child_structure = mutate_structure(child_structure, mutation_rate_structure)
#             new_population_struct.append(child_structure)
#         fitness_scores_struct_temp = best_so_far_paring_individual(new_population_struct,pop_brain,eval_pop_1=True)

#         # Generate new population brain
#         while len(new_population_brain) < pop_brain_sz:
#             parent1, parent2 = tournament_selection(pop_brain, fitness_scores_brain, tournament_sz)
#             child_weights = crossover_weights(parent1, parent2)
#             child_weights = mutate_weights(child_weights, mutation_rate_weights)
#             new_population_brain.append(child_weights)

#         fitness_scores_brain_temp = best_so_far_paring_individual(pop_struct,new_population_brain,eval_pop_1=False)

#         fitness_scores_struct = fitness_scores_struct_temp
#         fitness_scores_brain = fitness_scores_brain_temp
#         pop_brain = new_population_brain
#         pop_struct = new_population_struct

#         best_fitness_struct = max(fitness_scores_struct)
#         avg_fitness_struct = np.mean(fitness_scores_struct)
#         print(
#             f"Generation {generation + 1}: Structure - Best Fitness = {best_fitness_struct}, Average Fitness = {avg_fitness_struct}")
#         best_fitness_brain = max(fitness_scores_brain)
#         avg_fitness_brain = np.mean(fitness_scores_brain)
#         print(
#             f"Generation {generation + 1}: Brain - Best Fitness = {best_fitness_brain}, Average Fitness = {avg_fitness_brain}")

#         results[generation] = [generation+1,fitness_scores_struct[0],np.mean(fitness_scores_struct),fitness_scores_brain[0],np.mean(fitness_scores_brain)]

#     final_pairings = []
#     for structure in pop_struct:
#         for brain in pop_brain:
#             final_pairings.append((structure,brain))
#     with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#         fitness = np.array(list(executor.map(evaluate_fitness, *zip(*final_pairings))))

#     best_robot = final_pairings[np.argmax(fitness)][0]
#     best_weights = final_pairings[np.argmax(fitness)][1]
#     best_fitness = max(fitness)

#     results[nr_gens] = [nr_gens+1,best_fitness,np.mean(fitness),best_fitness,np.mean(fitness)]

#     results = pd.DataFrame(results,columns=['Iteration','Best Reward Struct','Mean Reward Struct','Best Reward Brain','Mean Reward Brain'])
#     results['Iteration'] = results['Iteration'].astype('int32')

#     return best_robot, best_weights, best_fitness, results

# def coevolution_v3(pop_struct_sz, pop_brain_sz, nr_gens, mutation_rate_structure, mutation_rate_weights, elitism_sz, tournament_sz):
#     """Run the coevolutionary algorithm."""
#     def mutate_structure(robot_structure, mutation_rate):
#         """Mutate the robot structure."""
#         mutated = copy.deepcopy(robot_structure)
#         for i in range(mutated.shape[0]):
#             for j in range(mutated.shape[1]):
#                 if random.random() < mutation_rate:
#                     mutated[i][j] = random.choice(VOXEL_TYPES)
#         while not is_connected(mutated):
#             mutated = copy.deepcopy(robot_structure)
#             for i in range(mutated.shape[0]):
#                 for j in range(mutated.shape[1]):
#                     if random.random() < mutation_rate:
#                         mutated[i][j] = random.choice(VOXEL_TYPES)
#         return mutated

#     def mutate_weights(weights, mutation_rate):
#         """Mutate the controller weights."""
#         mutated_weights = [copy.deepcopy(w) for w in weights]
#         for i, array in enumerate(mutated_weights):
#             mutation_mask = np.random.rand(*array.shape) < mutation_rate
#             noise = np.random.randn(*array.shape) * 0.02  # Mutation strength
#             mutated_weights[i] += mutation_mask * noise
#         return mutated_weights

#     def crossover_structure(parent1,parent2):
#         # Flatten parents to 1D arrays
#         parent1_flat = copy.deepcopy(parent1).flatten()
#         parent2_flat = copy.deepcopy(parent2).flatten()

#         mask = np.random.randint(0, 2, size=25)
#         child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

#         i = 1
#         while not is_connected(child_flat.reshape((5,5))):
#             #print("Wasn't Connected Crossover")
#             if i>5:
#                 child_flat = create_random_robot()
#                 break
#             mask = np.random.randint(0, 2, size=25)
#             child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
#             i+=1

#         # Reshape back to (5,5)
#         child = child_flat.reshape((5,5))

#         return child

#     def crossover_weights(parent1, parent2):
#         """Perform crossover on controller weights."""
#         child = [copy.deepcopy(w1) for w1 in parent1]
#         for i in range(len(child)):
#             mask = np.random.rand(*child[i].shape) < 0.5
#             child[i][mask] = parent2[i][mask]
#         return child
#         # return parent1

#     def tournament_selection(pop, fitness_scores, k):
#         """Select two parents using tournament selection."""
#         indices = np.random.choice(len(pop), k, replace=False)
#         sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
#         return pop[sorted_indices[0]], pop[sorted_indices[1]]

#     def random_paring(pop_s,pop_b):
#         fitness_w1 = []
#         pairings = []

#         # Evaluation of pop_w1
#         for w1 in pop_s:
#             partner = random.choice(pop_b) # choose randomly the partner
#             pairings.append((w1,partner))
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w1 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         fitness_w2 = []
#         pairings = []

#         # Evaluation of pop_w2
#         for w2 in pop_b:
#             partner = random.choice(pop_s) # choose randomly the partner
#             pairings.append((partner,w2))
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness_w2 = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         return fitness_w1,fitness_w2

#     def roulette_wheel_individual(pop_s, pop_b, fitness_pop_not_evaled, eval_pop_1=True):
#         pairings = []

#         if eval_pop_1:
#             eval_targets = pop_b
#         else:
#             eval_targets = pop_s

#         # Shift fitness values to make them non-negative
#         fitness_pop_not_evaled = np.clip(fitness_pop_not_evaled,a_min=0,a_max=np.inf)

#         # Avoid division by zero or all-zero fitness
#         total_fitness = np.sum(fitness_pop_not_evaled)
#         if total_fitness == 0:
#             probabilities = np.ones(len(eval_targets)) / len(eval_targets)
#         else:
#             probabilities = fitness_pop_not_evaled / total_fitness

#         # Now do roulette-based selection
#         selected_partners = random.choices(eval_targets, weights=probabilities, k=len(pop_s if eval_pop_1 else pop_b))

#         if eval_pop_1:
#             pairings = list(zip(pop_s, selected_partners))
#         else:
#             pairings = list(zip(selected_partners, pop_b))

#         # Evaluate fitness of these pairings
#         with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#             fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))

#         return fitness

#     def best_so_far_paring_individual(pop_s,pop_b,eval_pop_1=True):
#         pairings = []

#         if eval_pop_1:
#             for w1 in pop_s:
#                 partner = pop_b[0]
#                 pairings.append((w1,partner))
#             with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#                 fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))
#         else:
#             for w2 in pop_b:
#                 partner = pop_s[0]
#                 pairings.append((partner,w2))
#             with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#                 fitness = np.array(list(executor.map(evaluate_fitness, *zip(*pairings))))


#         return fitness

#     results = np.zeros((nr_gens+1,5))

#     pop_struct = [create_random_robot() for _ in range(pop_struct_sz)]
#     pop_brain = [create_random_weights(create_random_robot()) for _ in range(pop_brain_sz)]

#     fitness_scores_struct, fitness_scores_brain = random_paring(pop_struct,pop_brain)

#     for generation in range(nr_gens):
#         new_population_struct = []
#         new_population_brain = []
#         sorted_indices_struct = np.argsort(fitness_scores_struct)[::-1]
#         sorted_indices_brain = np.argsort(fitness_scores_brain)[::-1]
#         pop_struct = [pop_struct[i] for i in sorted_indices_struct]
#         fitness_scores_struct = [fitness_scores_struct[i] for i in sorted_indices_struct]
#         pop_brain = [pop_brain[i] for i in sorted_indices_brain]
#         fitness_scores_brain = [fitness_scores_brain[i] for i in sorted_indices_brain]

#         # Elitism
#         new_population_struct = list(copy.deepcopy(pop_struct[0:ELITISM_SIZE]))
#         new_population_brain = list(copy.deepcopy(pop_brain[0:ELITISM_SIZE]))

#         # Generate new population struct
#         while len(new_population_struct) < pop_struct_sz:
#             parent1, parent2 = tournament_selection(pop_struct, fitness_scores_struct, tournament_sz)
#             child_structure = crossover_structure(parent1, parent2)
#             child_structure = mutate_structure(child_structure, mutation_rate_structure)
#             new_population_struct.append(child_structure)
#         fitness_scores_struct_temp = roulette_wheel_individual(new_population_struct,pop_brain,fitness_scores_brain,eval_pop_1=True)

#         # Generate new population brain
#         while len(new_population_brain) < pop_brain_sz:
#             parent1, parent2 = tournament_selection(pop_brain, fitness_scores_brain, tournament_sz)
#             child_weights = crossover_weights(parent1, parent2)
#             child_weights = mutate_weights(child_weights, mutation_rate_weights)
#             new_population_brain.append(child_weights)
#         fitness_scores_brain_temp = roulette_wheel_individual(pop_struct,new_population_brain,fitness_scores_struct,eval_pop_1=False)

#         fitness_scores_struct = fitness_scores_struct_temp
#         fitness_scores_brain = fitness_scores_brain_temp
#         pop_brain = new_population_brain
#         pop_struct = new_population_struct

#         best_fitness_struct = max(fitness_scores_struct)
#         avg_fitness_struct = np.mean(fitness_scores_struct)
#         print(
#             f"Generation {generation + 1}: Structure - Best Fitness = {best_fitness_struct}, Average Fitness = {avg_fitness_struct}")
#         best_fitness_brain = max(fitness_scores_brain)
#         avg_fitness_brain = np.mean(fitness_scores_brain)
#         print(
#             f"Generation {generation + 1}: Brain - Best Fitness = {best_fitness_brain}, Average Fitness = {avg_fitness_brain}")

#         results[generation] = [generation+1,fitness_scores_struct[0],np.mean(fitness_scores_struct),fitness_scores_brain[0],np.mean(fitness_scores_brain)]

#     final_pairings = []
#     for structure in pop_struct:
#         for brain in pop_brain:
#             final_pairings.append((structure,brain))
#     with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
#         fitness = np.array(list(executor.map(evaluate_fitness, *zip(*final_pairings))))

#     best_robot = final_pairings[np.argmax(fitness)][0]
#     best_weights = final_pairings[np.argmax(fitness)][1]
#     best_fitness = max(fitness)

#     results[nr_gens] = [nr_gens+1,best_fitness,np.mean(fitness),best_fitness,np.mean(fitness)]

#     results = pd.DataFrame(results,columns=['Iteration','Best Reward Struct','Mean Reward Struct','Best Reward Brain','Mean Reward Brain'])
#     results['Iteration'] = results['Iteration'].astype('int32')

#     return best_robot, best_weights, best_fitness, results

# def coevolution(pop_sz, nr_gens, mutation_rate_structure, mutation_rate_weights, elitism_sz, tournament_sz):
#     """Run the coevolutionary algorithm."""
#     def mutate_structure(robot_structure, mutation_rate):
#         """Mutate the robot structure."""
#         mutated = copy.deepcopy(robot_structure)
#         for i in range(mutated.shape[0]):
#             for j in range(mutated.shape[1]):
#                 if random.random() < mutation_rate:
#                     mutated[i][j] = random.choice(VOXEL_TYPES)
#         while not is_connected(mutated):
#             mutated = copy.deepcopy(robot_structure)
#             for i in range(mutated.shape[0]):
#                 for j in range(mutated.shape[1]):
#                     if random.random() < mutation_rate:
#                         mutated[i][j] = random.choice(VOXEL_TYPES)
#         return mutated

#     def mutate_weights(weights, mutation_rate):
#         """Mutate the controller weights."""
#         mutated_weights = [w.copy() for w in weights]
#         for i, array in enumerate(mutated_weights):
#             mutation_mask = np.random.rand(*array.shape) < mutation_rate
#             noise = np.random.randn(*array.shape) * 0.02  # Mutation strength
#             mutated_weights[i] += mutation_mask * noise
#         return mutated_weights

#     def crossover_structure(parent1, parent2):
#         """Perform crossover on robot structures."""
#         child = copy.deepcopy(parent1)
#         for i in range(parent1.shape[0]):
#             for j in range(parent1.shape[1]):
#                 if random.random() < 0.5:
#                     child[i][j] = parent2[i][j]
#         while not is_connected(child):
#             child = crossover_structure(parent1, parent2)
#         return child

#     def crossover_weights(parent1, parent2):
#         """Perform crossover on controller weights."""
#         child = [w1.copy() for w1 in parent1]
#         for i in range(len(child)):
#             mask = np.random.rand(*child[i].shape) < 0.5
#             child[i][mask] = parent2[i][mask]
#         return child

#     def tournament_selection(pop, fitness_scores, k):
#         """Select two parents using tournament selection."""
#         indices = np.random.choice(len(pop), k, replace=False)
#         sorted_indices = sorted(indices, key=lambda idx: fitness_scores[idx], reverse=True)
#         return pop[sorted_indices[0]], pop[sorted_indices[1]]

#     pop = [(create_random_robot(), create_random_weights()) for _ in range(pop_sz)]
#     fitness_scores = [evaluate_fitness(robot, weights) for robot, weights in pop]

#     for generation in range(nr_gens):
#         new_population = []
#         sorted_indices = np.argsort(fitness_scores)[::-1]
#         pop = [pop[i] for i in sorted_indices]
#         fitness_scores = [fitness_scores[i] for i in sorted_indices]

#         # Elitism
#         new_population.extend(pop[:elitism_sz])

#         # Generate new population
#         while len(new_population) < pop_sz:
#             parent1, parent2 = tournament_selection(pop, fitness_scores, tournament_sz)
#             child_structure = crossover_structure(parent1[0], parent2[0])
#             child_weights = crossover_weights(parent1[1], parent2[1])
#             child_structure = mutate_structure(child_structure, mutation_rate_structure)
#             child_weights = mutate_weights(child_weights, mutation_rate_weights)
#             new_population.append((child_structure, child_weights))

#         pop = new_population
#         fitness_scores = [evaluate_fitness(robot, weights) for robot, weights in pop]

#         best_fitness = max(fitness_scores)
#         avg_fitness = np.mean(fitness_scores)
#         print(
#             f"Generation {generation + 1}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

#     best_robot, best_weights = pop[np.argmax(fitness_scores)]
#     return best_robot, best_weights


def run_coevolution(seed):
    json_file = dict()

    # Parameters
    pop_size_struct = 10
    pop_size_brain = 18
    num_generations_struct = 8
    num_generations_brain = 15
    mutation_rate_structure = 0.1
    mutation_rate_weights = 0.3
    mutation_strength_weights = 0.03
    elitism_size = 2
    tournament_size = 5

    print('Starting Coevolution...')
    start_time = time.time()

    best_robot, best_weights, best_fitness, results = coevolution_v6(
        pop_size_struct,
        pop_size_brain,
        num_generations_struct,
        num_generations_brain,
        mutation_rate_structure,
        mutation_rate_weights,
        mutation_strength_weights,
        elitism_size,
        tournament_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{seed}'
    path = os.path.join('students', 'results', 'all', 'coevolution', time_str)
    os.makedirs(path)

    json_file['elapsed_time'] = elapsed_time

    print("Elapsed time: " + str(elapsed_time))

    results.to_csv(path+'\\fitness_history.csv', index=False)

    print("Best robot structure found:")
    json_file['best_robot'] = best_robot.tolist()
    print(best_robot)
    print("Best fitness score:")
    json_file['best fitness'] = best_fitness
    print(best_fitness)

    json_file['total_eval'] = (pop_size_struct-2)*num_generations_struct*((pop_size_brain-2)*num_generations_brain+pop_size_brain) + pop_size_struct*((pop_size_brain-2)*num_generations_brain + pop_size_brain)

    with open(path+'\\data.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)

    connectivity = get_full_connectivity(best_robot)
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=best_robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, best_weights)
    create_gif_brain(
        best_robot,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,)

    return


def run_coevolution_cma(seed):
    json_file = dict()

    # Parameters
    pop_size_struct = 12
    pop_size_brain = 10
    num_generations_struct = 5
    num_generations_brain = 15
    mutation_rate_structure = 0.15
    tournament_size = 5

    print('Starting CMA Coevolution...')
    start_time = time.time()

    best_robot, best_weights, best_fitness, results = coevolution_cma(
        pop_size_struct,
        pop_size_brain,
        num_generations_struct,
        num_generations_brain,
        mutation_rate_structure,
        tournament_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    time_str = f'{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_S{seed}'
    path = os.path.join('students', 'results', 'all', 'cma_coevolution', time_str)
    os.makedirs(path)

    json_file['elapsed_time'] = elapsed_time

    print("Elapsed time: " + str(elapsed_time))

    results.to_csv(path+'\\fitness_history.csv', index=False)

    print("Best robot structure found:")
    json_file['best_robot'] = best_robot.tolist()
    print(best_robot)
    print("Best fitness score:")
    json_file['best fitness'] = best_fitness
    print(best_fitness)

    json_file['total_eval'] = (pop_size_struct-2)*num_generations_struct*((pop_size_brain-2)*num_generations_brain+pop_size_brain) + pop_size_struct*((pop_size_brain-2)*num_generations_brain + pop_size_brain)

    with open(path+'\\data.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)

    connectivity = get_full_connectivity(best_robot)
    env = gym.make(SCENARIO, max_episode_steps=STEPS,
                   body=best_robot, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)
    set_weights(brain, best_weights)
    create_gif_brain(
        best_robot,
        brain,
        filename=f'{path}\\best_robot.gif',
        scenario=SCENARIO,
        steps=STEPS,)

    return


def main():
    for seed in SEEDS:
        set_seed(seed)
        # run_coevolution(seed)
        run_coevolution_cma(seed)
    return


if __name__ == '__main__':
    main()
