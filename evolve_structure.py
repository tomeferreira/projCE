import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import os
import time


# ---- PARAMETERS ----
NUM_CORES = 4
NUM_GENERATIONS = 50  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait

def evaluate_fitness(robot_structure, view=False):    
    try:
        if not is_connected(robot_structure):
            return -10.0

        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0


def create_random_robot():
    """Generate a valid random robot structure."""
    
    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot

def hill_climber(iterations=1000):

    results = np.zeros((NUM_GENERATIONS,2))

    robot = create_random_robot()
    best_robot = robot
    best_fitness = evaluate_fitness(best_robot)

    print(best_robot)

    for it in range(NUM_GENERATIONS):
        

        robot = np.copy(best_robot)
        robot[random.randint(0,robot.shape[0]-1),random.randint(0,robot.shape[1]-1)] = random.choice(VOXEL_TYPES)
        while not is_connected(robot):
            robot = np.copy(best_robot)
            robot[random.randint(0,robot.shape[0]-1),random.randint(0,robot.shape[1]-1)] = random.choice(VOXEL_TYPES)
        print(robot)
        fitness_score = evaluate_fitness(robot)

        if fitness_score > best_fitness:
            best_fitness = fitness_score
            best_robot = robot
        
        print(f"Iteration {it + 1}: Fitness = {fitness_score}")

        results[it] = [it+1,fitness_score]
    
    results = pd.DataFrame(results,columns=['Iteration','Fitness'])
    results['Iteration'] = results['Iteration'].astype('int32')

    return best_robot, best_fitness, results

def run_hill_climber():
    
    # Example usage
    best_robot, best_fitness, results = hill_climber()
    
    time_str = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))+"_S"+str(STEPS)+"_G"+str(NUM_GENERATIONS)
    path = os.path.join("students","results","hill_climber",time_str)

    os.makedirs(path)

    results.to_csv(path+'\\fitness_history.csv',index=False)

    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    print(best_fitness)
    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        i += 1
    utils.create_gif(best_robot, filename=path+'\hill_climber.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

    return

def genetic_algorithm(population_size=20, mutation_rate=0.05):

    def mutate(robot, mutation_rate):
        # TODO YOUR CODE HERE
        for i in range(robot.shape[0]):
            for j in range(robot.shape[1]):
                if random.random()<=mutation_rate:
                    robot[i][j]=random.choice(VOXEL_TYPES)
        return robot
    
    def crossover(parent1, parent2):
        # Flatten parents to 1D arrays
        parent1_flat = parent1.flatten()
        parent2_flat = parent2.flatten()

        crossover_points = sorted(np.random.choice(25, 3, replace=False))

        # Perform three-point crossover
        child_flat = np.concatenate([
            np.copy(parent1_flat[:crossover_points[0]]),
            np.copy(parent2_flat[crossover_points[0]:crossover_points[1]]),
            np.copy(parent1_flat[crossover_points[1]:crossover_points[2]]),
            np.copy(parent2_flat[crossover_points[2]:])
        ])

        # Reshape back to (5,5)
        child = child_flat.reshape((5,5))

        return child

    def tournament_selection(population, k=5):
        # TODO YOUR CODE HERE
        # Note: use sorted(<...>, key=fitness) to sort by value of the fitness function call
        indices = np.random.choice(len(population),k,replace=False)
        ind = sorted(indices)[:2]
        return population[ind[0]],population[ind[1]]

    results = np.zeros((NUM_GENERATIONS,3))

    population = [create_random_robot() for _ in range(population_size)]

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        fitness_scores = np.array(list(executor.map(evaluate_fitness, population)))

    sorted_indices = np.argsort(-fitness_scores)
    population = [population[i] for i in sorted_indices]

    for it in range(NUM_GENERATIONS):

        new_population = list(np.copy(population[0:2]))
        while len(new_population) < population_size:
            p1, p2 = tournament_selection(population,10)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            fitness_scores = np.array(list(executor.map(evaluate_fitness, new_population)))
        sorted_indices = np.argsort(-fitness_scores)
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        population = [new_population[i] for i in sorted_indices]

        print(fitness_scores)
        print(f"Iteration {it + 1}: Best Fitness = {fitness_scores[0]} Average Fitness = {np.mean(fitness_scores)}")

        results[it] = [it+1,fitness_scores[0],np.mean(fitness_scores)]
    
    best_robot = population[0]
    best_fitness = fitness_scores[0]

    results = pd.DataFrame(results,columns=['Iteration','Fitness','Mean Fitness'])
    results['Iteration'] = results['Iteration'].astype('int32')

    return best_robot, best_fitness, results

def run_genetic_algorithm():

    # Example usage
    best_robot, best_fitness, results = genetic_algorithm()
    
    time_str = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))+"_S"+str(STEPS)+"_G"+str(NUM_GENERATIONS)
    path = os.path.join("students","results","ga",time_str)

    os.makedirs(path)

    results.to_csv(path+'\\fitness_history.csv',index=False)

    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    print(best_fitness)
    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        i += 1
    utils.create_gif(best_robot, filename=path+'\genetic_algorithm.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

    return

def main():
    start_time = time.time()
    # run_hill_climber()
    run_genetic_algorithm()
    end_time = time.time()
    print("Elapsed time: " + str(end_time-start_time))
    return

if __name__ == '__main__':
    main()