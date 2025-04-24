import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np

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

def main():
    graph_5_runs('students\\results\\ga\\just_reward',100,50,0.05,5,2)

    
    return


if __name__ == '__main__':
	main()