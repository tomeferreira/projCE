import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import pandas as pd
import numpy as np

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import os
import time


# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
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

    print(results.shape)

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

def main():
    
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

if __name__ == '__main__':
    main()