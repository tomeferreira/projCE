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

# ---- PARAMETERS ----
NUM_CORES = 6
NUM_GENERATIONS = 50  # Number of generations to evolve
ELITISM_SIZE = 2
TOURNAMENT_SIZE = 5
POP_SIZE = 50
MUTATION_RATE = 0.05
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait

robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

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
            speed_x = speed[0]
            speed_y = speed[1]   
            print(f'Speed X: {speed_x}')
            print(f'Speed Y: {speed_y}')
            pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)
            print(f'Pos: {pos}')
            orientation = sim.object_orientation_at_time(sim.get_time(),'robot')
            print(f'Orientation: {orientation}')

            print('\n\n')

            print(robot_structure[4])

            time.sleep(0.5)

            fitness+=(speed_x-3*abs(speed_y))/STEPS

            # if orientation<math.pi/8 or orientation>15*math.pi/8:
            #     fitness+=1/STEPS
            # else:
            #     fitness-=1/STEPS
            
            if terminated or truncated:
                end_pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)[0]
                env.reset()
                break

        x_speeds = np.sort(x_speeds)[::-1]
        fitness = np.mean(x_speeds[0:20])

        if end_pos==None:
            end_pos = end_pos = np.mean(sim.object_pos_at_time(sim.get_time(),'robot'),axis=1)[0]

        if end_pos<start_pos:
            return t_reward,fitness-100

        # print(f'Speed sum: {fitness}')
        # print(f'Pos: {end_pos-start_pos}')

        # fitness+= (end_pos-start_pos)

        viewer.close()
        env.close()
        return t_reward,fitness
    except (ValueError, IndexError) as e:
        return 0.0,0.0


def main():
    
    print(evaluate_fitness(robot_structure,view=True))

    return

if __name__ == '__main__':
    main()