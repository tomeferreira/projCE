import plotly.graph_objects as go
import pandas as pd

'''
Código do ChatGPT
'''

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