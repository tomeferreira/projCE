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
NUM_CORES = 9
NUM_GENERATIONS = 250  # Number of generations to evolve
ELITISM_SIZE = 2
TOURNAMENT_SIZE = 5
POP_SIZE = 50
MUTATION_RATE = 0.05
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'BridgeWalker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = sinusoidal_wave

def circular_variance(thetas):
    C = np.mean(np.cos(thetas))
    S = np.mean(np.sin(thetas))
    R = np.sqrt(C**2 + S**2)
    V = 1 - R
    return V

def evaluate_fitness(robot_structure, view=False):    
    try:
        fitness = 0
        end_pos = None
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        start_pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)[0]
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        x_speeds = np.zeros(STEPS)
        orientations = np.zeros(STEPS)
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            # fitness += reward

            speed = np.mean(sim.object_vel_at_time(sim.get_time(),'robot'),axis=1)
            x_speeds[t]=speed[0] 
            orientations[t] = sim.object_orientation_at_time(sim.get_time(),'robot')

            if terminated or truncated:
                end_pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)[0]
                env.reset()
                break
        
        # x_speeds = np.sort(x_speeds)
        # # Nunca vi speed >100, logo divido por 100 para "Normalizar"
        # fitness += np.mean(x_speeds[0:40])/100
        # x_speeds = x_speeds[::-1]
        # fitness += np.mean(x_speeds[0:40])/100

        mean_speed = np.mean(x_speeds)
        fitness+=mean_speed/100

        # print(f'Speed fitness: {fitness}')

        orientation_variance = np.abs(circular_variance(orientations))
        # print(f'Orientation fitness: {orientation_variance*10}')
        fitness-=orientation_variance*10

        if end_pos==None:
            end_pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)[0]

        if end_pos<start_pos:
            fitness-=100
        else:
            # print(f'Distancia fitness: {(end_pos-start_pos)/100}')
            fitness+=(end_pos-start_pos)/100

        # if np.count_nonzero(robot_structure[4])==5:
        #     fitness-=10

        # print(f'Multiplier: {np.count_nonzero(robot_structure)/25}')
        # fitness=fitness*(1-(np.count_nonzero(robot_structure)/25)*0.25)

        viewer.close()
        env.close()
        return t_reward,fitness
    except (ValueError, IndexError) as e:
        return 0.0,0.0


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

def random_search():
    """Perform a random search to find the best robot structure."""

    best_robot = None
    best_fitness = -float('inf')

    population = [create_random_robot() for _ in range(POP_SIZE*(NUM_GENERATIONS+1))]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        temp_results = np.array(list(executor.map(evaluate_fitness, population)))
    
    reward_scores = temp_results[:,0]
    fitness_scores = temp_results[:,1]

    sorted_indices = np.argsort(-reward_scores)
    population = [population[i] for i in sorted_indices]
    reward_scores = [reward_scores[i] for i in sorted_indices]
    fitness_scores = [fitness_scores[i] for i in sorted_indices]

    best_robot = population[0]
    best_fitness = fitness_scores[0]
    best_reward = reward_scores[0]

    return best_robot, best_fitness, best_reward

def run_random_search(seed):

    json_file = dict()

    # Example usage
    best_robot, best_fitness, best_reward = random_search()
    
    time_str = str(time.strftime("%Y-%m-%d_%Hh%Mm%Ss"))+"_S"+str(seed)+"_G"+str(NUM_GENERATIONS)
    path = os.path.join("students","results","random_search",time_str)

    os.makedirs(path)

    print("Best robot structure found:")
    json_file['best_robot']=best_robot.tolist()
    print(best_robot)
    print("Best fitness score:")
    json_file['best fitness']=best_fitness
    print(best_fitness)
    print("Best reward score:")
    json_file['best reward']=best_reward
    print(best_reward)

    json_file['total_eval']=(NUM_GENERATIONS+1)*POP_SIZE

    with open(path+'\\data.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)

    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        i += 1
    utils.create_gif(best_robot, filename=path+'\\random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

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
    
    def uniform_crossover(parent1,parent2):
        # Flatten parents to 1D arrays
        parent1_flat = copy.deepcopy(parent1).flatten()
        parent2_flat = copy.deepcopy(parent2).flatten()

        mask = np.random.randint(0, 2, size=25)
        child_flat = np.where(mask == 1, parent1_flat, parent2_flat)

        i = 1
        while not is_connected(child_flat.reshape((5,5))):
            #print("Wasn't Connected Crossover")
            if i>5:
                child_flat = create_random_robot()
                break
            mask = np.random.randint(0, 2, size=25)
            child_flat = np.where(mask == 1, parent1_flat, parent2_flat)
            i+=1

        # Reshape back to (5,5)
        child = child_flat.reshape((5,5))

        return child

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

        # Reshape back to (5,5)
        child = child_flat.reshape((5,5))

        return child

    def tournament_selection(population, k=5):
        # TODO YOUR CODE HERE
        # Note: use sorted(<...>, key=fitness) to sort by value of the fitness function call
        indices = np.random.choice(len(population),k,replace=False)
        ind = sorted(indices)[:2]
        return population[ind[0]],population[ind[1]]

    results = np.zeros((NUM_GENERATIONS,5))

    population = [create_random_robot() for _ in range(POP_SIZE)]

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        temp_results = np.array(list(executor.map(evaluate_fitness, population)))

    reward_scores = temp_results[:,0]
    fitness_scores = temp_results[:,1]

    sorted_indices = np.argsort(-reward_scores)
    population = [population[i] for i in sorted_indices]

    for it in range(NUM_GENERATIONS):

        new_population = list(copy.deepcopy(population[0:ELITISM_SIZE]))
        while len(new_population) < POP_SIZE:
            p1, p2 = tournament_selection(population,TOURNAMENT_SIZE)
            child = uniform_crossover(p1, p2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
        
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            temp_results = np.array(list(executor.map(evaluate_fitness, new_population)))

        reward_scores = temp_results[:,0]
        fitness_scores = temp_results[:,1]
        
        sorted_indices = np.argsort(-reward_scores)
        reward_scores = [reward_scores[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        population = [new_population[i] for i in sorted_indices]

        print(fitness_scores)
        print(np.array(fitness_scores)-np.array(reward_scores))
        print(f"Iteration {it + 1}: Best Fitness = {fitness_scores[0]} Average Fitness = {np.mean(fitness_scores)} Best Reward = {max(reward_scores)} Mean Reward = {np.mean(reward_scores)}")

        results[it] = [it+1,fitness_scores[0],np.mean(fitness_scores),max(reward_scores),np.mean(reward_scores)]
    
    best_robot = population[0]
    best_fitness = fitness_scores[0]
    best_reward = reward_scores[0]

    results = pd.DataFrame(results,columns=['Iteration','Best Fitness','Mean Fitness','Best Reward','Mean Fitness'])
    results['Iteration'] = results['Iteration'].astype('int32')

    return best_robot, best_fitness, best_reward, results

def run_genetic_algorithm(seed):

    json_file = dict()

    start_time = time.time()

    # Example usage
    best_robot, best_fitness, best_reward, results = genetic_algorithm()
    
    end_time = time.time()
    elapsed_time = end_time-start_time

    json_file['Controller']='sinusoidal_wave'

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
    print("Best reward score:")
    json_file['best reward']=best_reward
    print(best_reward)

    json_file['total_eval']=NUM_GENERATIONS*POP_SIZE

    with open(path+'\\data.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=4)

    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
        i += 1
    utils.create_gif(best_robot, filename=path+'\genetic_algorithm.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

    return

def main():
    for seed in SEEDS:
        set_seed(seed)
        run_genetic_algorithm(seed)
    return

if __name__ == '__main__':
    main()