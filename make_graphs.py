import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import glob

'''
Código do ChatGPT
'''

def graph_1_run():
    # Load the CSV file
    df = pd.read_csv("students\\results\\evolotionary_algorithm\\2025-03-28_19h05m46s_G50_P20_M0.05_T5_E2\\fitness_history.csv")

    # Create the figure
    fig = go.Figure()

    # Add the best overall reward line
    fig.add_trace(go.Scatter(
        x=df["Iteration"], 
        y=df["Fitness"], 
        mode='lines', 
        name='Best Overall Reward',
        line=dict(color='blue')
    ))

    # Add the mean reward line
    fig.add_trace(go.Scatter(
        x=df["Iteration"], 
        y=df["Mean Fitness"], 
        mode='lines', 
        name='Mean Reward',
        line=dict(color='orange')
    ))

    # Customize layout
    fig.update_layout(
        title='',
        xaxis_title='Generation',
        yaxis_title='Reward',
        legend=dict(x=0.7, y=0.1)
    )

    # Show the figure
    fig.show()

    return

def graph_5_runs(PATH,NUM_GENERATIONS,POP_SIZE,MUTATION_RATE,TOURNAMENT_SIZE,ELITISM_SIZE):
    string = f'G{NUM_GENERATIONS}_P{POP_SIZE}_M{MUTATION_RATE}_T{TOURNAMENT_SIZE}_E{ELITISM_SIZE}'

    df_list=[]

    dir_list = os.listdir(PATH)
    for file in dir_list:
        if string in file:
            df = pd.read_csv(PATH+'\\'+file+'\\fitness_history.csv')
            df_list.append(df)
    
    mean_array = np.mean([df.values for df in df_list], axis=0)
    mean_df = pd.DataFrame(mean_array, columns=df_list[0].columns)

    std_array = np.std([df.values for df in df_list], axis=0)
    std_df = pd.DataFrame(std_array, columns=df_list[0].columns)

    # Create the figure
    fig = go.Figure()

    # Add the best overall reward line
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"], 
        y=mean_df["Fitness"], 
        mode='lines', 
        name='Best Overall Reward',
        line=dict(color='blue')
    ))

    # Add the mean reward line
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"], 
        y=mean_df["Mean Fitness"], 
        mode='lines', 
        name='Mean Reward',
        line=dict(color='orange')
    ))

    # Calculate upper and lower bounds for mean ± std
    upper_bound_mean = mean_df["Mean Fitness"] + std_df["Mean Fitness"]
    lower_bound_mean = mean_df["Mean Fitness"] - std_df["Mean Fitness"]

    # Add the upper bound (invisible trace, used for fill)
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"],
        y=upper_bound_mean,
        mode='lines',
        line=dict(width=0),
        name='Upper Bound Mean',
        showlegend=False
    ))

    # Add the lower bound with fill to the previous trace
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"],
        y=lower_bound_mean,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',  # Fill to trace before (upper bound)
        fillcolor='rgba(255,165,0,0.2)',  # Light orange
        name='Std Dev Mean',
        showlegend=False
    ))

     # Calculate upper and lower bounds for mean ± std
    upper_bound_best = mean_df["Fitness"] + std_df["Fitness"]
    lower_bound_best = mean_df["Fitness"] - std_df["Fitness"]

    # Add the upper bound (invisible trace, used for fill)
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"],
        y=upper_bound_best,
        mode='lines',
        line=dict(width=0),
        name='Upper Bound Best',
        showlegend=False
    ))

    # Add the lower bound with fill to the previous trace
    fig.add_trace(go.Scatter(
        x=mean_df["Iteration"],
        y=lower_bound_best,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',  # Fill to trace before (upper bound)
        fillcolor='rgba(0,0,255,0.2)',  # Light blue
        name='Std Dev Best',
        showlegend=False
    ))

    # Customize layout
    fig.update_layout(
        title='',
        xaxis_title='Generation',
        yaxis_title='Reward',
        legend=dict(x=0.7, y=0.1)
    )

    # Show the figure
    fig.show()

    return


def plot_fitness_evolution(algorithm_paths, save_path="graphs", history_column="Best Fitness"):
    """
    Plot the fitness evolution for multiple algorithms, averaging across multiple runs, using Matplotlib.
    
    Args:
        algorithm_paths (dict): A dictionary where keys are algorithm names and values are lists of file paths to fitness history CSVs.
        save_path (str): The directory where the plot will be saved.
        history_column (str): The column name to plot (e.g., "Best Fitness" or "Mean Fitness").
    """
    plt.figure(figsize=(12, 6))

    for algorithm, paths in algorithm_paths.items():
        fitness_data = []

        # Load and process each file for the algorithm
        for file in paths:
            df = pd.read_csv(file)
            fitness_data.append(df[history_column].values)

        # Calculate the mean and standard deviation of fitness evolution across all files
        fitness_df = pd.DataFrame(fitness_data)
        mean_fitness = fitness_df.mean(axis=0)
        std_fitness = fitness_df.std(axis=0)

        # Plot the mean fitness evolution line
        plt.plot(range(1, len(mean_fitness) + 1), mean_fitness, label=algorithm)

        # Add the "shadow" (mean ± std)
        plt.fill_between(
            range(1, len(mean_fitness) + 1),
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            alpha=0.2  # Transparency for the shadow
        )

    # Customize the plot
    plt.title(f'Best Reward Evolution Across Algorithms')
    plt.xlabel('Iteration')
    plt.ylabel('Best Reward')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)

    # Save the figure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"best_reward_evolution.png"
    plt.savefig(f"{save_path}/{file_name}", dpi=300)

    # Show the plot
    plt.show()


def main():
    # graph_5_runs('students\\results\\ga\\just_reward',100,50,0.05,5,2)

    downstepper_3_2 = {
        "CMA-ES (Sigma = 1.3)": glob.glob("final_results/3_2/DownStepper/cma_big_sigma/*/fitness_history.csv"),
        "CMA-ES (Sigma = 0.5)": glob.glob("final_results/3_2/DownStepper/cma_small_sigma/*/fitness_history.csv"),
        "DE (base aleatória)": glob.glob("final_results/3_2/DownStepper/de_random_base/*/fitness_history.csv"),
        "DE (melhor base)": glob.glob("final_results/3_2/DownStepper/de_best_base/*/fitness_history.csv"),
        "GA (menos mutação)": glob.glob("final_results/3_2/DownStepper/ga_less_mutation/*/fitness_history.csv"),
        "GA (mais mutação)": glob.glob("final_results/3_2/DownStepper/ga_more_mutation/*/fitness_history.csv"),
        "Random Search": glob.glob("final_results/3_2/DownStepper/random/*/fitness_history.csv"),
    }

    obstacle_traverser_3_2 = {
        "CMA-ES (Sigma = 1.3)": glob.glob("final_results/3_2/ObstacleTraverser/cma/*/fitness_history.csv"),
        "DE (melhor base)": glob.glob("final_results/3_2/ObstacleTraverser/de/*/fitness_history.csv"),
    }

    gapjumper_3_3 = {
        "CMA-ES (Sigma = 2.5)": glob.glob("final_results/3_3/GapJumper/CMAES_Sigma2_5/*/fitness_history.csv"),
        "CMA-ES (Sigma = 3.5)": glob.glob("final_results/3_3/GapJumper/CMAES_Sigma3_5/*/fitness_history.csv"),
    }

    # Plot the fitness evolution
    plot_fitness_evolution(obstacle_traverser_3_2, "graphs/3_2/ObstacleTraverser", "Best Fitness")

    return


if __name__ == '__main__':
	main()