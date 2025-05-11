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

def one_way_ind_anova(data):
    """
    parametric
    many samples
    independent
    """
    F,pval = st.f_oneway(*data)
    return (F,pval)

def t_test_ind(data1,data2, eq_var=True):
    """
    parametric
    two samples
    independent
    """
    t,pval = st.ttest_ind(data1,data2, equal_var=eq_var)
    return (t,pval)

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
	print(f'ANOVA test: {one_way_ind_anova((best_alternating_gait,best_hopping_motion,best_sinusoidal_wave))}')

	print(f'Alternating Gait vs Hopping Motion: {t_test_ind(best_alternating_gait,best_hopping_motion)}')
	print(f'Alternating Gait vs Sinusoidal Wave: {t_test_ind(best_alternating_gait, best_sinusoidal_wave)}')

	print(f'Hopping Motion vs Sinusoidal Wave: {t_test_ind(best_hopping_motion, best_sinusoidal_wave)}')

	# Which is better?

	print('Of the similar ones, which one is better?')
	print(f'Alternating Gait: {np.mean(best_alternating_gait)}')
	print(f'Hopping Motion: {np.mean(best_hopping_motion)}')
	print(f'Sinusoidal Wave: {np.mean(best_sinusoidal_wave)}')

	return

def test_3_1_Walker_different_fitness():
	# Load data
	reward = load_data('students\\final_results\\3_1\\250_GA\\uniform_crossover_5_mutation')
	fitness = load_data('students\\final_results\\3_1\\my_fitness')

	print(reward)
	print(fitness)

	# Test Normality
	print('Normality testing w\\Shapiro-Wilk')
	print(f'Reward: {test_normal_sw(reward)}')
	print(f'Fitness: {test_normal_sw(fitness)}')

	print(f'Reward vs Fitness: {t_test_ind(reward,fitness)}')

	# Which is better?

	print('Of the similar ones, which one is better?')
	print(f'Reward: {np.mean(reward)}')
	print(f'Fitness: {np.mean(fitness)}')

	return

def test_3_1_BridgeWalker_different_controller():
	# Load data
	best_alternating_gait = load_data('students\\final_results\\3_1\\BridgeWalker_alternating_gait')
	best_hopping_motion = load_data('students\\final_results\\3_1\\BridgeWalker_hopping_motion')
	best_sinusoidal_wave = load_data('students\\final_results\\3_1\\BridgeWalker_sinusoidal_waves')

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

def test_3_2_downstepper():
    # Load data for each algorithm
    best_random = load_data('final_results/3_2/DownStepper/random', ranking_method='best fitness')
    best_ga_less_mutation = load_data('final_results/3_2/DownStepper/ga_less_mutation', ranking_method='best fitness')
    best_cma_big_sigma = load_data('final_results/3_2/DownStepper/cma_big_sigma', ranking_method='best fitness')
    best_de_random_base = load_data('final_results/3_2/DownStepper/de_random_base', ranking_method='best fitness')
    best_ga_more_mutation = load_data('final_results/3_2/DownStepper/ga_more_mutation', ranking_method='best fitness')
    best_cma_small_sigma = load_data('final_results/3_2/DownStepper/cma_small_sigma', ranking_method='best fitness')
    best_de_best_base = load_data('final_results/3_2/DownStepper/de_best_base', ranking_method='best fitness')

    # Print the loaded data
    print('Best Random Search:', best_random)
    print('Best Genetic Algorithm (less mutation):', best_ga_less_mutation)
    print('Best CMA-ES (big sigma):', best_cma_big_sigma)
    print('Best Differential Evolution (random base):', best_de_random_base)
    print('Best Genetic Algorithm (more mutation):', best_ga_more_mutation)
    print('Best CMA-ES (small sigma):', best_cma_small_sigma)
    print('Best Differential Evolution (best base):', best_de_best_base)

    # Test Normality using Shapiro-Wilk
    print('\nNormality testing w/Shapiro-Wilk')
    print(f'    Random: {test_normal_sw(best_random)}')
    print(f'    GA (less mutation): {test_normal_sw(best_ga_less_mutation)}')
    print(f'    CMA-ES (big sigma): {test_normal_sw(best_cma_big_sigma)}')
    print(f'    DE (random base): {test_normal_sw(best_de_random_base)}')
    print(f'    GA (more mutation): {test_normal_sw(best_ga_more_mutation)}')
    print(f'    CMA-ES (small sigma): {test_normal_sw(best_cma_small_sigma)}')
    print(f'    DE (best base): {test_normal_sw(best_de_best_base)}')

    # Perform ANOVA
    anova_result = st.f_oneway(
        best_random,
        best_ga_less_mutation,
        best_cma_big_sigma,
        best_de_random_base,
        best_ga_more_mutation,
        best_cma_small_sigma,
        best_de_best_base
    )
    print(f'\nANOVA test: F={anova_result.statistic:.4f}, p={anova_result.pvalue:.8f}')

    # Perform T-Test for pairwise comparisons if ANOVA is significant
    if anova_result.pvalue < 0.05:
        print('ANOVA is significant, which means there is a difference between the groups.')
        print('Performing pairwise T-Tests for comparisons...\n')

        # Define all groups and their labels
        groups = {
            "Random Search": best_random,
            "GA (less mutation)": best_ga_less_mutation,
            "CMA-ES (big sigma)": best_cma_big_sigma,
            "DE (random base)": best_de_random_base,
            "GA (more mutation)": best_ga_more_mutation,
            "CMA-ES (small sigma)": best_cma_small_sigma,
            "DE (best base)": best_de_best_base
        }

        # Perform pairwise T-Tests
        group_names = list(groups.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                group1 = group_names[i]
                group2 = group_names[j]
                t_stat, p_value = st.ttest_ind(groups[group1], groups[group2])
                print(f'    {group1} vs {group2}: t={t_stat:.4f}, p={p_value:.8f}')

    # Print mean values
    print('\nMean Performance:')
    for name, data in groups.items():
        print(f'    {name}: {np.mean(data):.4f}')

    # The t-tests did not show significant differences, so we can't conclude which algorithm is better based on the means alone


def test_3_2_obstacle_traverser():
    best_cma = load_data('final_results/3_2/ObstacleTraverser/cma', ranking_method='best fitness')
    best_de = load_data('final_results/3_2/ObstacleTraverser/de', ranking_method='best fitness')

    # Print the loaded data
    print('Best CMA-ES:', best_cma)
    print('Best Differential Evolution:', best_de)
    
    # Test Normality using Shapiro-Wilk
    print('\nNormality testing w/Shapiro-Wilk')
    print(f'    CMA-ES: {test_normal_sw(best_cma)}')
    print(f'    DE: {test_normal_sw(best_de)}')

    # The data is normal: use T-Test. No need for ANOVA, as we only have two groups
    t_stat, p_value = st.ttest_ind(best_cma, best_de)
    print(f'\nT-Test: t={t_stat:.4f}, p={p_value:.8f}')

    # Print mean values to determine which is better
    print('\nMean Performance:')
    print(f'    CMA-ES: {np.mean(best_cma):.4f}')
    print(f'    DE: {np.mean(best_de):.4f}')

    # Find and print the best algorithm
    means = [('CMA-ES', np.mean(best_cma)), ('DE', np.mean(best_de))]
    best_algo = max(means, key=lambda x: x[1])
    print(f'\nBest performing algorithm: {best_algo[0]} with mean fitness {best_algo[1]:.4f}')

def test_3_3_gapjumper():
    best_cmaes_sigma3_5 = load_data('final_results/3_3/GapJumper/CMAES_Sigma3_5', ranking_method='best fitness')
    best_cmaes_sigma2_5 = load_data('final_results/3_3/GapJumper/CMAES_Sigma2_5', ranking_method='best fitness')

    # Print the loaded data
    print('Best CMA-ES (sigma 3.5):', best_cmaes_sigma3_5)
    print('Best CMA-ES (sigma 2.5):', best_cmaes_sigma2_5)

    # Test Normality using Shapiro-Wilk
    print('\nNormality testing w/Shapiro-Wilk')
    print(f'    CMA-ES (sigma 3.5): {test_normal_sw(best_cmaes_sigma3_5)}')
    print(f'    CMA-ES (sigma 2.5): {test_normal_sw(best_cmaes_sigma2_5)}')

    # p is close to being smaller than 0.05, but the data is normal: use T-Test. No need for ANOVA, as we only have two groups

    t_stat, p_value = st.ttest_ind(best_cmaes_sigma3_5, best_cmaes_sigma2_5)
    print(f'\nT-Test: t={t_stat:.4f}, p={p_value:.8f}')

    # Print mean values
    print('\nMean Performance:')
    print(f'    CMA-ES (sigma 3.5): {np.mean(best_cmaes_sigma3_5):.4f}')
    print(f'    CMA-ES (sigma 2.5): {np.mean(best_cmaes_sigma2_5):.4f}')

    # We cannot conclude which algorithm is better based on the means alone, as the T-Test did not show significant differences.

def test_3_3_cavecrawler():
    best_cmaes_sigma3_5 = load_data('students/final_results/3_3/CaveCrawler/CMAES_Sigma3_5', ranking_method='best fitness')
    best_cmaes_sigma2_5 = load_data('students\\final_results\\3_3\\CaveCrawler_v0', ranking_method='best fitness')

    # Print the loaded data
    print('Best CMA-ES (sigma 3.5):', best_cmaes_sigma3_5)
    print('Best CMA-ES (sigma 2.5):', best_cmaes_sigma2_5)

    # Test Normality using Shapiro-Wilk
    print('\nNormality testing w/Shapiro-Wilk')
    print(f'    CMA-ES (sigma 3.5): {test_normal_sw(best_cmaes_sigma3_5)}')
    print(f'    CMA-ES (sigma 2.5): {test_normal_sw(best_cmaes_sigma2_5)}')

    # p is close to being smaller than 0.05, but the data is normal: use T-Test. No need for ANOVA, as we only have two groups

    t_stat, p_value = st.ttest_ind(best_cmaes_sigma3_5, best_cmaes_sigma2_5)
    print(f'\nT-Test: t={t_stat:.4f}, p={p_value:.8f}')

    # Print mean values
    print('\nMean Performance:')
    print(f'    CMA-ES (sigma 3.5): {np.mean(best_cmaes_sigma3_5):.4f}')
    print(f'    CMA-ES (sigma 2.5): {np.mean(best_cmaes_sigma2_5):.4f}')

def main():

	# test_3_1_Walker_same_controller()

	# test_3_1_Walker_different_controller()
    
	# test_3_1_Walker_different_fitness()

	# test_3_1_BridgeWalker_different_controller()

	# test_3_2_downstepper()

    # test_3_2_obstacle_traverser()

	# test_3_3_gapjumper()
      
	test_3_3_cavecrawler()

	return

if __name__ == '__main__':
	main()