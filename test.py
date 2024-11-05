import pandas as pd
import numpy as np

# Probability array
prob = [0.7, 0.3]  # Example probabilities, where the first value is for team1 and the second value is for team2

# Probability of choosing team1
p_team1 = prob[0]

# Number of simulations
num_simulations = 10000  # Adjust this value as needed

# Simulate selections
team1_selections = np.random.choice(['team1', 'team2'], size=num_simulations, p=prob)

# Count occurrences of team1
team1_count = np.sum(team1_selections == 'team1')

# Calculate percentage of team1 selections
team1_percentage = (team1_count / num_simulations) * 100

print(f"Team1 is chosen approximately {team1_percentage:.2f}% of the time.")
