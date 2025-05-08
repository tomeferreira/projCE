import json
import scipy.stats as st
import os
import numpy as np

def load_data(path, ranking_method='best reward'):
	bests = []

	dir_list = os.listdir(path)
	for dir in dir_list:
		file_path = os.path.join(path, dir, 'data.json')
		with open(file_path, 'r') as f:
			data = json.load(f)
			bests.append(data[ranking_method])
		
	return sorted(bests)

def test_normal_sw(data):
    """Shapiro-Wilk"""
    norm_data = (data - np.mean(data))/(np.std(data)/np.sqrt(len(data)))
    return st.shapiro(norm_data)

def kruskal_wallis(data):
    """
    non parametric
    many samples
    independent
    """     
    H,pval = st.kruskal(*data)
    return (H,pval)

def mann_whitney(data1,data2):
    """
    non parametric
    two samples
    independent
    """    
    return st.mannwhitneyu(data1, data2)

def test_3_1_Walker_same_controller():
	# Load data
	best_random = load_data('students\\results\\random_search')
	best_ga_3point_5_mutation = load_data('students\\results\\ga\\250_GA\\3point_crossover_5_mutation')
	best_ga_3point_10_mutation = load_data('students\\results\\ga\\250_GA\\3point_crossover_10_mutation')
	best_ga_uniform_5_mutation = load_data('students\\results\\ga\\250_GA\\uniform_crossover_5_mutation')
	best_ga_uniform_10_mutation = load_data('students\\results\\ga\\250_GA\\uniform_crossover_10_mutation')

	print(best_random)
	print(best_ga_3point_5_mutation)
	print(best_ga_3point_10_mutation)
	print(best_ga_uniform_5_mutation)
	print(best_ga_uniform_10_mutation)

	# Test Normality
	print('Normality testing w\\Shapiro-Wilk')
	print(f'Random: {test_normal_sw(best_random)}')
	print(f'GA w\\3_point_cross & 0.05% mut: {test_normal_sw(best_ga_3point_5_mutation)}')
	print(f'GA w\\3_point_cross & 0.1% mut: {test_normal_sw(best_ga_3point_10_mutation)}')
	print(f'GA w\\uniform_cross & 0.05% mut: {test_normal_sw(best_ga_uniform_5_mutation)}')
	print(f'GA w\\uniform_cross & 0.1% mut: {test_normal_sw(best_ga_uniform_10_mutation)}')

	# Hypothesis testing
	print(f'Kruskal Wallis test: {kruskal_wallis((best_random,best_ga_3point_5_mutation,best_ga_3point_10_mutation,best_ga_uniform_5_mutation,best_ga_uniform_10_mutation))}')

	print(f'Random vs GA w\\3_point_cross & 0.05% mut: {mann_whitney(best_random,best_ga_3point_5_mutation)}')
	print(f'Random vs GA w\\3_point_cross & 0.1% mut: {mann_whitney(best_random, best_ga_3point_10_mutation)}')
	print(f'Random vs GA w\\uniform_cross & 0.05% mut: {mann_whitney(best_random, best_ga_uniform_5_mutation)}')
	print(f'Random vs GA w\\uniform_cross & 0.1% mut: {mann_whitney(best_random, best_ga_uniform_10_mutation)}')

	print(f'GA w\\3_point_cross & 0.05% mut vs GA w\\3_point_cross & 0.1% mut: {mann_whitney(best_ga_3point_5_mutation, best_ga_3point_10_mutation)}')
	print(f'GA w\\3_point_cross & 0.05% mut vs GA w\\uniform_cross & 0.05% mut: {mann_whitney(best_ga_3point_5_mutation, best_ga_uniform_5_mutation)}')
	print(f'GA w\\3_point_cross & 0.05% mut vs GA w\\uniform_cross & 0.1% mut: {mann_whitney(best_ga_3point_5_mutation, best_ga_uniform_10_mutation)}')

	print(f'GA w\\3_point_cross & 0.1% mut vs GA w\\uniform_cross & 0.05% mut: {mann_whitney(best_ga_3point_10_mutation, best_ga_uniform_5_mutation)}')
	print(f'GA w\\3_point_cross & 0.1% mut vs GA w\\uniform_cross & 0.1% mut: {mann_whitney(best_ga_3point_10_mutation, best_ga_uniform_10_mutation)}')

	print(f'GA w\\uniform_cross & 0.05% mut vs GA w\\uniform_cross & 0.1% mut: {mann_whitney(best_ga_uniform_5_mutation, best_ga_uniform_10_mutation)}')


	# Which is better?

	print('Of the similar ones, which one is better?')
	print(f'GA w\\3_point_cross & 0.05% mut: {np.mean(best_ga_3point_5_mutation)}')
	print(f'GA w\\3_point_cross & 0.1% mut: {np.mean(best_ga_3point_10_mutation)}')
	print(f'GA w\\uniform_cross & 0.05% mut: {np.mean(best_ga_uniform_5_mutation)}')
	print(f'GA w\\uniform_cross & 0.1% mut: {np.mean(best_ga_uniform_10_mutation)}')

	return

def test_3_1_Walker_different_controller():
	# Load data
	best_alternating_gait = load_data('students\\results\\ga\\250_GA\\uniform_crossover_5_mutation')
	best_hopping_motion = load_data('students\\results\\ga\\hopping_motion')
	best_sinusoidal_wave = load_data('students\\results\\ga\\sinusoidal_wave')

	print(best_alternating_gait)
	print(best_hopping_motion)
	print(best_sinusoidal_wave)

	# Test Normality
	print('Normality testing w\\Shapiro-Wilk')
	print(f'Alternating Gait: {test_normal_sw(best_alternating_gait)}')
	print(f'Hopping Motion: {test_normal_sw(best_hopping_motion)}')
	print(f'Sinusoidal Wave: {test_normal_sw(best_sinusoidal_wave)}')

	# Hypothesis testing
	print(f'Kruskal Wallis test: {kruskal_wallis((best_alternating_gait,best_hopping_motion,best_sinusoidal_wave))}')

	print(f'Alternating Gait vs Hopping Motion: {mann_whitney(best_alternating_gait,best_hopping_motion)}')
	print(f'Alternating Gait vs Sinusoidal Wave: {mann_whitney(best_alternating_gait, best_sinusoidal_wave)}')

	print(f'Hopping Motion vs Sinusoidal Wave: {mann_whitney(best_hopping_motion, best_sinusoidal_wave)}')

	# Which is better?

	print('Of the similar ones, which one is better?')
	print(f'Alternating Gait: {np.mean(best_alternating_gait)}')
	print(f'Hopping Motion: {np.mean(best_hopping_motion)}')
	print(f'Sinusoidal Wave: {np.mean(best_sinusoidal_wave)}')

	return

def test_3_2_walker():
    # Load data for each algorithm
    best_random = load_data('results/controller/first_map/random', ranking_method='best fitness')
    best_ga = load_data('results/controller/first_map/ga', ranking_method='best fitness')
    best_cma = load_data('results/controller/first_map/cma', ranking_method='best fitness')
    best_de = load_data('results/controller/first_map/de', ranking_method='best fitness')

    # Print the loaded data
    print("Random Search:", best_random)
    print("Genetic Algorithm:", best_ga)
    print("CMA-ES:", best_cma)
    print("Differential Evolution:", best_de)

    # Test Normality
    print('\nNormality testing w\\Shapiro-Wilk')
    print(f'Random Search: {test_normal_sw(best_random)}')
    print(f'Genetic Algorithm: {test_normal_sw(best_ga)}')
    print(f'CMA-ES: {test_normal_sw(best_cma)}')
    print(f'Differential Evolution: {test_normal_sw(best_de)}')

    # Kruskal-Wallis test for overall difference
    print(f'\nKruskal Wallis test: {kruskal_wallis((best_random, best_ga, best_cma, best_de))}')

    # Pairwise comparisons using Mann-Whitney U test
    print('\nPairwise Comparisons:')
    print(f'Random Search vs Genetic Algorithm: {mann_whitney(best_random, best_ga)}')
    print(f'Random Search vs CMA-ES: {mann_whitney(best_random, best_cma)}')
    print(f'Random Search vs Differential Evolution: {mann_whitney(best_random, best_de)}')
    
    print(f'Genetic Algorithm vs CMA-ES: {mann_whitney(best_ga, best_cma)}')
    print(f'Genetic Algorithm vs Differential Evolution: {mann_whitney(best_ga, best_de)}')
    
    print(f'CMA-ES vs Differential Evolution: {mann_whitney(best_cma, best_de)}')

    # Print mean values to determine which is better
    print('\nMean Performance:')
    print(f'Random Search: {np.mean(best_random)}')
    print(f'Genetic Algorithm: {np.mean(best_ga)}')
    print(f'CMA-ES: {np.mean(best_cma)}')
    print(f'Differential Evolution: {np.mean(best_de)}')

    # Find and print the best algorithm
    means = [
        ("Random Search", np.mean(best_random)),
        ("Genetic Algorithm", np.mean(best_ga)),
        ("CMA-ES", np.mean(best_cma)),
        ("Differential Evolution", np.mean(best_de))
    ]
    best_algo = max(means, key=lambda x: x[1])
    print(f'\nBest performing algorithm: {best_algo[0]} with mean fitness {best_algo[1]:.4f}')
	


def main():

	#test_3_1_Walker_same_controller()

	#test_3_1_Walker_different_controller()
	
	test_3_2_walker()

	return

if __name__ == '__main__':
	main()