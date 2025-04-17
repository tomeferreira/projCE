import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import os
import time


SEEDS=[68167,91803,39233,58400,97664]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    print("Seed set")
    return

# ---- PARAMETERS ----
NUM_CORES = 6
NUM_GENERATIONS = 100  # Number of generations to evolve
ELITISM_SIZE = 2
TOURNAMENT_SIZE = 5
POP_SIZE = 50
MUTATION_RATE = 0.1
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait

def evaluate_fitness(robot_structure, view=False):    
    try:
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
        
        robot = copy.deepcopy(best_robot)
        robot[random.randint(0,robot.shape[0]-1),random.randint(0,robot.shape[1]-1)] = random.choice(VOXEL_TYPES)
        while not is_connected(robot):
            robot = copy.deepcopy(best_robot)
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

def genetic_algorithm():

    def mutate(robot, mutation_rate):
        mut_robot = copy.deepcopy(robot)
        for i in range(mut_robot.shape[0]):
            for j in range(mut_robot.shape[1]):
                if random.random()<=mutation_rate:
                    choice = random.choice(VOXEL_TYPES)
                    while mut_robot[i][j] == choice:
                        choice = random.choice(VOXEL_TYPES)
                    mut_robot[i][j]= choice
        while not is_connected(mut_robot):
            #print("Wasn't Connected Mutation")
            mut_robot = copy.deepcopy(robot)
            for i in range(mut_robot.shape[0]):
                for j in range(mut_robot.shape[1]):
                    if random.random()<=mutation_rate:
                        choice = random.choice(VOXEL_TYPES)
                        while mut_robot[i][j] == choice:
                            choice = random.choice(VOXEL_TYPES)
                        mut_robot[i][j]= choice
        return mut_robot
    
    def crossover_3points(parent1, parent2):
        # Flatten parents to 1D arrays
        parent1_flat = copy.deepcopy(parent1).flatten()
        parent2_flat = copy.deepcopy(parent2).flatten()

        crossover_points = sorted(np.random.choice(25, 3, replace=False))

        # Perform three-point crossover
        child_flat = np.concatenate([
            copy.deepcopy(parent1_flat[:crossover_points[0]]),
            copy.deepcopy(parent2_flat[crossover_points[0]:crossover_points[1]]),
            copy.deepcopy(parent1_flat[crossover_points[1]:crossover_points[2]]),
            copy.deepcopy(parent2_flat[crossover_points[2]:])
        ])

        i = 1
        while not is_connected(child_flat.reshape((5,5))):
            #print("Wasn't Connected Crossover")
            if i>5:
                child_flat = create_random_robot()
                break
            crossover_points = sorted(np.random.choice(25, 3, replace=False))
            child_flat = np.concatenate([
                copy.deepcopy(parent1_flat[:crossover_points[0]]),
                copy.deepcopy(parent2_flat[crossover_points[0]:crossover_points[1]]),
                copy.deepcopy(parent1_flat[crossover_points[1]:crossover_points[2]]),
                copy.deepcopy(parent2_flat[crossover_points[2]:])
            ])
            i+=1

        # if((child_flat==parent1_flat).all() or (child_flat==parent2_flat).all()):
        #     print("Child is a copy")

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

    population = [create_random_robot() for _ in range(POP_SIZE)]

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        fitness_scores = np.array(list(executor.map(evaluate_fitness, population)))

    sorted_indices = np.argsort(-fitness_scores)
    population = [population[i] for i in sorted_indices]

    for it in range(NUM_GENERATIONS):

        new_population = list(copy.deepcopy(population[0:ELITISM_SIZE]))
        while len(new_population) < POP_SIZE:
            p1, p2 = tournament_selection(population,TOURNAMENT_SIZE)
            child = crossover_3points(p1, p2)
            child = mutate(child, MUTATION_RATE)
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

def run_genetic_algorithm(seed):

    json_file = dict()

    start_time = time.time()

    # Example usage
    best_robot, best_fitness, results = genetic_algorithm()
    
    end_time = time.time()
    elapsed_time = end_time-start_time

    json_file['elapsed_time']=elapsed_time

    print("Elapsed time: " + str(elapsed_time))

    time_str = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))+"_S"+str(seed)+"_G"+str(NUM_GENERATIONS)+"_P"+str(POP_SIZE)+"_M"+str(MUTATION_RATE)+"_T"+str(TOURNAMENT_SIZE)+"_E"+str(ELITISM_SIZE)
    path = os.path.join("students","results","ga",time_str)

    os.makedirs(path)

    results.to_csv(path+'\\fitness_history.csv',index=False)

    print("Best robot structure found:")
    json_file['best_robot']=best_robot.tolist()
    print(best_robot)
    print("Best fitness score:")
    json_file['best fitness']=best_fitness
    print(best_fitness)

    json_file['total_eval']=NUM_GENERATIONS*POP_SIZE

    with open(path+'\\data.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)

    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        i += 1
    utils.create_gif(best_robot, filename=path+'\genetic_algorithm.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

    return

def main():
    for seed in SEEDS:
        set_seed(seed)
        # run_hill_climber()
        run_genetic_algorithm(seed)
    return

if __name__ == '__main__':
    main()