# Imports for general use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process

# Import from Log_reg model
# Imporing model related variables
from ncaabwins_logreg import log_reg, X, scaler

# Imporstatments from ncaabwinsmodel
# Importing dataframes
from ncaabwinsmodel import mteams, mseeds_df, mtournslot_df, kenpom_fin_df

# Imporing dictionaries
from ncaabwinsmodel import conf_mapping

# Importing functions
from ncaabwinsmodel import random_id, stat_swap

# Import from Seed Prediction
# Importing model
from seedprediction import knn_best, kenpom_knn_model_2024

# Load necessary data (team IDs, features, etc.)
teams_2024_raw = pd.read_csv('data/2024_tourney_seeds.csv')

# Creating a TeamID/Team Name dictionary
team_dict = dict(zip(mteams['TeamID'], mteams['Team']))

# Grabbing just the Mens teams (this is a model and sim for the mens tournament)
teams_2024_dfm = teams_2024_raw[teams_2024_raw['Tournament'] == 'M']

# Generate all possible pairs of teams
teams_pairs = []
for team1_id in teams_2024_dfm['TeamID']:
    for team2_id in teams_2024_dfm['TeamID']:
        if team1_id != team2_id:  # Exclude same team pairs
            teams_pairs.append((team1_id, team2_id))

# Get the index of the 'Season' column
season_index = kenpom_fin_df.columns.get_loc('Season')

# Move the 'Season' column to the first position
cols = list(kenpom_fin_df.columns)
cols.insert(0, cols.pop(season_index))
kenpom_fin_df = kenpom_fin_df[cols]

# Split the W-L column into two columns: W and L
kenpom_fin_df[['W', 'L']] = kenpom_fin_df['W-L'].str.split('-', expand=True)

# Convert both columns to numeric
kenpom_fin_df['W'] = pd.to_numeric(kenpom_fin_df['W'])
kenpom_fin_df['L'] = pd.to_numeric(kenpom_fin_df['L'])

# Drop old column
kenpom_fin_df = kenpom_fin_df.drop(columns = ['W-L'])

# Prepping Data for simulation, using only 2024 season
kenpom_fin_df = kenpom_fin_df[kenpom_fin_df['Season']==2024]

# Conference dictionary mapping for abbreviation
conf_mapping = {
    'SEC': 'sec',
    'CUSA': 'cusa',
    'MAC': 'mac',
    'B12': 'big_twelve',
    'B10': 'big_ten',
    'MWC': 'mwc',
    'BSky': 'big_sky',
    'ASun': 'a_sun',
    'MVC': 'mvc',
    'BE': 'big_east',
    'Horz': 'horizon',
    'OVC': 'ovc',
    'ACC': 'acc',
    'P10': 'pac_ten',
    'Slnd': 'southland',
    'A10': 'aac',
    'SB': 'sun_belt',
    'Ivy': 'ivy',
    'WCC': 'wcc',
    'WAC': 'wac',
    'CAA': 'caa',
    'Pat': 'patriot',
    'MAAC': 'maac',
    'NEC': 'nec',
    'AE': 'aec',
    'SWAC': 'swac',
    'MEAC': 'meac',
    'Sum': 'summit',
    'P12': 'pac_twelve',
    'Amer': 'americ_east'
}

# Apply conference mapping to Conf_A and Conf_B columns
kenpom_fin_df['Conf'] = kenpom_fin_df['Conf'].map(conf_mapping)

# Define conference tiers definition based on 2024 seeding results
conference_tiers = {
    'sec': 1, 'big_twelve': 1, 'big_ten': 1, 'acc': 1, 'big_east': 1, 'pac_twelve': 1, 'mwc': 1,
    'aac': 2, 'americ_east': 2, 'mvc': 2, 'wcc': 2
}

# Assign Tier 3 to the rest of the conferences
all_conferences = set(conf_mapping.values())
tier_three_conferences = all_conferences - set(conference_tiers.keys())
conference_tiers.update({conf: 3 for conf in tier_three_conferences})

# Create a function to map conferences to tiers
def map_tier(conf):
    return conference_tiers.get(conf, 3)  # Default to tier 3 if not found

# Apply conference tier mapping to Conf_A and Conf_B columns
kenpom_fin_df['Conf_Tier'] = kenpom_fin_df['Conf'].apply(map_tier)

# Create LabelEncoder object (for conferences)
label_encoder = LabelEncoder()

# Fit and transform Conf_A and Conf_B columns to integers
kenpom_fin_df['Conf'] = label_encoder.fit_transform(kenpom_fin_df['Conf'])

# Fill NAN values for seed
kenpom_fin_df['Seed'] = kenpom_fin_df['Seed'].fillna(99)

# This is a dummy variable is removed before any analysis is done
kenpom_fin_df['WinnerID'] = kenpom_fin_df['TeamID']

# Dictionaries created to be used later

# Dictionary Mapping Team to ID
team_to_id = dict(zip(kenpom_fin_df['Team'], kenpom_fin_df['TeamID']))

# Dictionary Mapping Team to Seed
team_to_seed = dict(zip(kenpom_fin_df['Team'], kenpom_fin_df['Seed']))

# Final Prep for data set that was modeled in previous ncaabwinsmodel
kenpom_fin2_df = kenpom_fin_df.copy()
kenpom_fin2_df = kenpom_fin2_df.drop(['FirstD1Season','LastD1Season','Luck','AdjEM'], axis = 1)

def prep_data_pred(team1_id, team2_id):
    # Function takes in two teams and returns a one row dataframe which is identical to the dataframe which the random
    # forest model from ncaabwinsmodel is trained on. To see its column names and order print(X.columns)

    # team1_id: id associated with team1
    # team2_id: id associated with team1

    # Getting the kenpom data for both Team1 and Team2 and renaming the columns to be the same as from the model
    row1 = kenpom_fin2_df[kenpom_fin2_df['TeamID'] == team1_id]
    row1.columns = [col + "_A" for col in row1.columns]
    row1 = row1.rename(columns = {'Season_A':'Season'})
    row1 = row1.rename(columns = {'WinnerID_A':'WinnerID'})
    row2 = kenpom_fin2_df[kenpom_fin2_df['TeamID'] == team2_id]
    row2.columns = [col + "_B" for col in row2.columns]
    # Dropping Season_B, not needed
    row2 = row2.drop(columns = ['Season_B'])
    row2 = row2.drop(columns = ['WinnerID_B'])

    # Combining the rows into row/df
    row1, row2 = row1.reset_index(drop = True), row2.reset_index(drop = True)
    stats_df = pd.concat([row1, row2],axis =1)

    # Reindex the DataFrame with the desired column order
    stats_df = stats_df.reindex(columns=X.columns)
    return stats_df

# Make predictions for each team pair
predictions = []

for team1_id, team2_id in teams_pairs:
    # Prepare features for prediction
    features = prep_data_pred(team1_id, team2_id)

    # Make prediction using the model
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
    prediction = log_reg.predict_proba(features_scaled)
    predictions.append((team1_id, team2_id, prediction))

# Format predictions into DataFrame
predictions_df = pd.DataFrame(predictions, columns=['TeamID1', 'TeamID2', 'Prediction'])

# Save predictions to a CSV file for possible debug and other uses
predictions_df.to_csv('data/_tournament_predictions.csv', index=False)


def prep_data_sim(seeds, preds):
    # seeds: Standard seeds csv provided by kaggle competetion, should only be for 2024 season
    # preds: prediciton dataframe generated by using the model making predictions after the prep_data_pred function has been used
    # based on current (11/04/2024) logistic regression model, data must be scaled.

    # Function preparing the data for the simulation
    seed_dict = seeds.set_index('Seed')['TeamID'].to_dict()
    inverted_seed_dict = {value: key for key, value in seed_dict.items()}
    probs_dict = {}

    # For loop creating probability dictionary (probs_dict)
    for team1, team2, probs in zip(preds['TeamID1'], preds['TeamID2'], preds['Prediction']):
        probs_dict.setdefault(team1, {})[team2] = 1 - probs[0]
        probs_dict.setdefault(team2, {})[team1] = probs[0]

    # Outputs

    # seed_dict: maps region specific seed to Team ID W01 = 1163,..., X01 = 1314 etc
    # inverted_seed_dict: same as seed dict but columns are flipped
    # probs_dict: dictionary with every matchup combination and their associated probabilities

    return seed_dict, inverted_seed_dict, probs_dict


preds = predictions_df

# Getting the seeds and tournament slots for the mens tournament in 2024
mseeds_df2024 = mseeds_df[mseeds_df['Season'] == 2024]
mtournslot_df2024 = mtournslot_df[mtournslot_df['Season'] == 2024]

# Getting the data from prep_data_sims
seed_dict, inverted_seed_dict, probs_dict = prep_data_sim(mseeds_df2024, preds)

# Saving used data to dfs and the to csvs to better understand the process of what is happening, this is not necessary
# for the code to run

# probs_dict
probs_df = pd.DataFrame.from_dict(probs_dict, orient='index')
probs_df.to_csv('data/_probs_dict.csv')

# seed_dict
seed_dict = pd.DataFrame.from_dict(seed_dict, orient='index')
seed_dict.to_csv('data/_seed_dict.csv')

# inverted seed_dict
inverted_seed_dict = pd.DataFrame.from_dict(inverted_seed_dict, orient='index')
inverted_seed_dict.to_csv('data/_inverted_seed_dict.csv')

# prediction dataframe
preds.to_csv('data/_preds.csv', index = False)

def simulate(round_slots, seeds, inverted_seeds, probs, random_values, sim=True):
    '''
    Simulates each round of the tournament.

    Parameters:
    - round_slots: DataFrame containing information on who is playing in each round.
    - seeds (dict): Dictionary mapping seed values to team IDs.
    - inverted_seeds (dict): Dictionary mapping team IDs to seed values.
    - probs (dict): Dictionary containing matchup probabilities.
    - random_values (array-like): Array with precomputed random-values.
    - sim (boolean): Simulates match if True. Chooses team with higher probability as winner otherwise.

    Returns:
    - list: List with winning team IDs for each match.
    - list: List with corresponding slot names for each match.
    '''
    winners = []
    slots = []
    for slot, strong, weak, random_val in zip(round_slots.Slot, round_slots.StrongSeed, round_slots.WeakSeed,
                                              random_values):
        team1, team2 = seeds[strong], seeds[weak]

        # Get the probability of team_1 winning
        prob = probs[team1][team2]


        if sim:
            # Randomly determine the winner based on the probability
            winner = np.random.choice([team1, team2], p=prob)

        else:
            # Determine the winner based on the higher probability
            winner = team1 if prob[0] > prob[1] else team2

        # Append the winner and corresponding slot to the lists
        winners.append(winner)
        slots.append(slot)

        seeds[slot] = winner

    # Convert winners to original seeds using the inverted_seeds dictionary
    return [inverted_seeds[w] for w in winners], slots

def run_simulation(brackets, seeds, preds, round_slots, sim):
    '''
    Runs a simulation of bracket tournaments.

    Parameters:
    - brackets (int): Number of brackets to simulate.
    - seeds (pd.DataFrame): DataFrame containing seed information.
    - preds (pd.DataFrame): DataFrame containing prediction information for each match-up.
    - round_slots (pd.DataFrame): DataFrame containing information about the tournament rounds.
    - sim (boolean): Simulates matches if True. Chooses team with higher probability as winner otherwise.

    Returns:
    - pd.DataFrame: DataFrame with simulation results.
    '''
    # Get relevant data for the simulation
    seed_dict, inverted_seed_dict, probs_dict = prep_data_sim(seeds, preds)

    # Lists to store simulation results
    results = []
    bracket = []
    slots = []

    # Precompute random-values
    random_values = np.random.random(size=(brackets, len(round_slots)))

    # Iterate through the specified number of brackets
    for b in tqdm(range(1, brackets + 1)):
        # Run single simulation
        r, s = simulate(round_slots, seed_dict, inverted_seed_dict, probs_dict, random_values[b - 1], sim)

        # Update results
        results.extend(r)
        bracket.extend([b] * len(r))
        slots.extend(s)

    # Create final DataFrame
    result_df = pd.DataFrame({'Bracket': bracket, 'Slot': slots, 'Team': results})

    return result_df


# Running the Sims!
result_m = run_simulation(brackets=10000, seeds=mseeds_df2024, preds=preds, round_slots=mtournslot_df2024, sim=True)

# Isolating Mens tournament
result_m['Tournament'] = 'M'

#Creating new df for submission
submission = result_m
submission.reset_index(inplace=True, drop=True)
submission.index.names = ['RowId']
submission = submission.drop(columns = 'Tournament')
submission = submission.rename(columns = {'Team':'Seed'})

# Saving submission file to examine what is happening
submission.to_csv('data/_submission.csv', index=False)

# Expanding on submission file so it is easier to read for humans
result_analysis_df = submission.merge(mseeds_df2024, on = 'Seed', how = 'inner')

# Dropping season and adding team ID for analysis on results
submission_with_team_info = result_analysis_df.drop(columns = 'Season')
submission_with_team_info = result_analysis_df.merge(mteams, on = 'TeamID', how = 'inner')
submission_with_team_info = submission_with_team_info.drop(['FirstD1Season','LastD1Season'], axis = 1)

# Saving new submmission
submission_with_team_info.to_csv('data/_result_analysis.csv', index=False)

# Calculating how many times each team makes it to X round
# Initialize dictionaries to store counts for each round
sweet_sixteen_counts = {}
elite_eight_counts = {}
final_four_counts = {}
championship_counts = {}

# Iterate over each row in the DataFrame
for index, row in submission_with_team_info.iterrows():
    # Get the team in this row
    team = row['Team']

    # Get the round of this match
    round_name = row['Slot'].split('_')[0]

    # Increment the count for the respective round for this team
    if 'R2' in round_name:
        sweet_sixteen_counts[team] = sweet_sixteen_counts.get(team, 0) + 1
    elif 'R3' in round_name:
        elite_eight_counts[team] = elite_eight_counts.get(team, 0) + 1
    elif 'R4' in round_name:
        final_four_counts[team] = final_four_counts.get(team, 0) + 1
    elif 'R6' in round_name:
        championship_counts[team] = championship_counts.get(team, 0) + 1

# Create DataFrames from the dictionaries
sweet_sixteen_df = pd.DataFrame(list(sweet_sixteen_counts.items()), columns=['Team', 'SweetSixteenCount'])
elite_eight_df = pd.DataFrame(list(elite_eight_counts.items()), columns=['Team', 'EliteEightCount'])
final_four_df = pd.DataFrame(list(final_four_counts.items()), columns=['Team', 'FinalFourCount'])
championship_df = pd.DataFrame(list(championship_counts.items()), columns=['Team', 'ChampionshipCount'])

# Merge all DataFrames on 'Team'
round_counts_df = pd.merge(sweet_sixteen_df, elite_eight_df, on='Team', how='outer')
round_counts_df = pd.merge(round_counts_df, final_four_df, on='Team', how='outer')
round_counts_df = pd.merge(round_counts_df, championship_df, on='Team', how='outer')

# General cleaning of round_counts
round_counts_df = round_counts_df.fillna(0)
round_counts_df = round_counts_df.merge(mteams, on = 'Team', how ='inner')
round_counts_df = round_counts_df.drop(columns = ['FirstD1Season','LastD1Season'])

# ave round_counts_df to a CSV file
round_counts_df.to_csv('data/_round_counts.csv', index=False)

# Define the labels (Teams and Rounds)
teams = list(round_counts_df.index)
round_labels = ['Sweet Sixteen', 'Elite 8', 'Final Four', 'Championship']

# Combine team names and round labels
labels = teams + round_labels

# Initialize source and target nodes (mapping rounds to teams)
sources = []
targets = []
values = []

# Iterate over each team to create the Sankey data
for idx, team in enumerate(teams):
    # Mapping Team -> Sweet Sixteen -> Elite 8 -> Final Four -> Final Game
    sources.extend([idx, idx, idx, idx])  # Source node index
    targets.extend([len(teams), len(teams) + 1, len(teams) + 2, len(teams) + 3])  # Target node index
    values.extend([
        round_counts_df.iloc[idx]['SweetSixteenCount'],
        round_counts_df.iloc[idx]['EliteEightCount'],
        round_counts_df.iloc[idx]['FinalFourCount'],
        round_counts_df.iloc[idx]['ChampionshipCount']
    ])

# Create the x, y positions for team labels on the left and round labels on the right
x_positions = [0.1] * len(teams) + [0.5, 0.6, 0.7, 0.8]  # Teams on the left, rounds more to the right
y_positions = [i / len(teams) for i in range(len(teams))] + [0.2, 0.4, 0.6, 0.8]  # Space the rounds vertically

# Create the Sankey diagram using Plotly
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,  # Padding between nodes
        thickness=10,  # Thickness of nodes
        line=dict(color="black", width=0.5),
        label=labels,  # All labels (teams + rounds)
        x=x_positions,
        y=y_positions
    ),
    link=dict(
        source=sources,  # The starting points (source nodes)
        target=targets,  # The end points (target nodes)
        value=values  # Values (frequencies) that connect sources to targets
    )
))

# Update layout and show plot
fig.update_layout(title_text="March Madness Team Advancement Flow", font_size=10, width=1000, height=900)
fig.show()

# Percentages for top 4 seeds for each region
top_16_finish_df = round_counts_df[['Team', 'SweetSixteenCount', 'EliteEightCount','FinalFourCount','ChampionshipCount']]
top_16_finish_df['Seed'] = top_16_finish_df['Team'].map(team_to_seed).astype(float)
top_seeds_finish_df = top_16_finish_df[top_16_finish_df['Seed'] <= 4]

# Basis for showing any Team(s) frequency in the tournament, can change to any roundand teams using same logic as
# currently use in top_seeds_finish_df

# Set up the Seaborn style
sns.set(style="whitegrid")

# Create the barplot with 'Team' on the x-axis and 'SweetSixteenCount' on the y-axis
plt.figure(figsize=(10, 6))  # Adjust the figure size
barplot = sns.barplot(data=top_seeds_finish_df, x='Team', y='SweetSixteenCount', palette='Blues_d')

# Add labels and title
plt.xlabel("Team", fontsize=12)
plt.ylabel("Sweet Sixteen Count", fontsize=12)
plt.title("Number of Sweet Sixteen Appearances by Team", fontsize=14)

# Rotate x-axis labels for better readability if needed
plt.xticks(rotation=45, ha="right")

# Show the plot
plt.tight_layout()
plt.show()

#Top Champs

# Set up the Seaborn style
sns.set(style="whitegrid")

top_champs_df = round_counts_df.sort_values('ChampionshipCount', ascending = False)[:20]

# Create the barplot with 'Team' on the x-axis and 'SweetSixteenCount' on the y-axis
plt.figure(figsize=(10, 6))  # Adjust the figure size
barplot = sns.barplot(data=top_champs_df, x='Team', y='ChampionshipCount', palette='husl')

# Add labels and title
plt.xlabel("Team", fontsize=12)
plt.ylabel("Total Count", fontsize=12)
plt.title("Championship Wins", fontsize=14)

# Rotate x-axis labels for better readability if needed
plt.xticks(rotation=45, ha="right")

# Show the plot
plt.tight_layout()
plt.show()

# Getting Mean, Median and Mode wins by team

# Get unique teams and brackets
teams = submission_with_team_info['Team'].unique()
brackets = submission_with_team_info['Bracket'].unique()

# Create a MultiIndex for all possible combinations of teams and brackets
multi_index = pd.MultiIndex.from_product([teams, brackets], names=['Team', 'Bracket'])

# Group by 'Team' and 'Bracket' to count occurrences
team_counts = submission_with_team_info.groupby(['Team', 'Bracket']).size().reset_index(name='count')

# Reindex to include all combinations of Team and Bracket, filling missing values with 0
win_counts_complete_df = team_counts.set_index(['Team', 'Bracket']).reindex(multi_index, fill_value=0).reset_index()

# Now, group by 'Team' and calculate mean, median, and mode
win_mean_df = win_counts_complete_df.groupby('Team').mean(numeric_only=True)
win_median_df = win_counts_complete_df.groupby('Team').median(numeric_only=True)
win_mode_df = win_counts_complete_df.groupby('Team').agg(lambda x: x.mode()[0])

# Drop the 'Bracket' column since it's not needed in the aggregation results
win_mean_df = win_mean_df.drop(['Bracket'], axis=1)
win_median_df = win_median_df.drop(['Bracket'], axis=1)
win_mode_df = win_mode_df.drop(['Bracket'], axis=1)

# win_mean_df.sort_values('count', ascending = False)[:20]

# Creating dataframe for average wins by seed from https://bracketodds.cs.illinois.edu/seedadv.html
avg_wins_seed = {
    'Seed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'Average Wins': [3.30, 2.33, 1.84, 1.56, 1.15, 1.04, .9, .71, .62, .60, .67, .51, .25, .16, .1, .013]
}

avg_wins_seed_df = pd.DataFrame(avg_wins_seed)

# Adding seed column to win_mean_df
win_mean_df = win_mean_df.reset_index()
win_mean_df['Seed'] = win_mean_df['Team'].map(team_to_seed)
# win_mean_df

# Merging, Renaming dfs to create new df for Expectations column
# Merge
win_mean_df = win_mean_df.merge(avg_wins_seed_df, how = 'inner', on = 'Seed')

# Renaming columns
win_mean_df.rename(columns = {'count':'Simulated Average Wins','Average Wins':'Average Historical Seed Wins'}, inplace = True)

# New Column
win_mean_df['Wins Over Seed Expectation'] = win_mean_df['Simulated Average Wins'] - win_mean_df['Average Historical Seed Wins']

# This can be turned into a function and used for any df of the (median or mode in this case) and the performance ratio can be
# set to any number, in this case anything with 25% of win total is within expectation
# It should be noted that Expectation is only dependent/determined by seed

# Loop through each row of the DataFrame
for index, row in win_mean_df.iterrows():

    # Calculate the ratio of Sim_Seed_Performance to Average Historical Seed Wins
    performance_ratio = row['Wins Over Seed Expectation'] / row['Average Historical Seed Wins']

    # Assign 'Expectations' based on the performance ratio
    if performance_ratio >= .25:
        win_mean_df.at[index, 'Expectations'] = 'Overperformer'
    elif performance_ratio <= -.25:
        win_mean_df.at[index, 'Expectations'] = 'Underperformer'
    else:
        win_mean_df.at[index, 'Expectations'] = 'Average'

performance_sort_df = win_mean_df.sort_values('Wins Over Seed Expectation', ascending = False)
# performance_sort_df

# Number of Average, Under and Overperformers, output muted
# performance_sort_df.value_counts('Expectations')

# Sorted df for teams on average overperforming based strictly on seed expectation. Output muted
overperformers_df = performance_sort_df[performance_sort_df['Expectations'] == 'Overperformer']
# overperformers_df

# Sorted df for teams on average underperforming based strictly on seed expectation. Output muted
underperformers_df = performance_sort_df[performance_sort_df['Expectations'] == 'Underperformer']
# underperformers_df

#Plot for the Biggest underperformers with respect to seed expectation

#Sort for biggest Underperformers
underperformers_df = underperformers_df.sort_values('Wins Over Seed Expectation', ascending = True)
# Concatenate Team and Seed to create new labels for the x-axis
underperformers_df['Team (Seed)'] = underperformers_df['Team'] + ' ' + underperformers_df['Seed'].astype(str)

# Create a bar plot using Seaborn
sns.barplot(x = 'Team (Seed)', y = 'Wins Over Seed Expectation', data = underperformers_df[:10], palette='coolwarm')

# Add labels and title
plt.xlabel('Team', fontsize=12)
plt.ylabel('Projected Wins Over Seed Expectation', fontsize=12)
plt.title('Projected Underperformers', fontsize=14)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.show()

# Plot for the Biggest Overperformers with respect to seed expectation

# Concatenate Team and Seed to create new labels for the x-axis
overperformers_df['Team (Seed)'] = overperformers_df['Team'] + ' ' + overperformers_df['Seed'].astype(str)
# Create a bar plot using Seaborn
sns.barplot(x = 'Team (Seed)',y = 'Wins Over Seed Expectation', data=overperformers_df, palette='coolwarm')

# Add labels and title
plt.xlabel('Team', fontsize=12)
plt.ylabel('Projected Wins Over Seed Expectation', fontsize=12)
plt.title('Projected Overperformers', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Display the plot
plt.tight_layout()
plt.show()

# Final Four Combinations

# Preperations for plotting most common Final Fours (any round) and checking how often any combination of
# teams make a certain round

# Filter to Rows for Teams in Final Four
# Can be changed to R3 for Elite 8 R5 for Championship game, etc
final_four_df = submission_with_team_info[submission_with_team_info['Slot'].str.contains('R4')]

# Groupby Final Four and make list
final_four_teams = final_four_df.groupby('Bracket')['Team'].apply(list).reset_index()

# Sort the teams in each tuple to make sure order does not matter in count
final_four_teams['Final Four'] = final_four_teams['Team'].apply(lambda x: tuple(sorted(x)))

# Apply value counts
final_four_value_counts = final_four_teams['Final Four'].value_counts()


# Function to check for any combination of 1 to 4 teams and plot the matching Final Four combinations
def check_final_four_combination(combination, df, top_x):
    '''
    combination = specific combination to search
    df = dataframe being searche, in this case final four combinations
    top_x = top x combos to graph
    '''

    # Sort the input combination to ensure order doesn't matter
    sorted_combination = tuple(sorted(combination))

    # Initialize a list to store matching combinations and their counts
    matched_combinations = []

    # Initialize a counter to track total occurrences of the combination
    total_count = 0
    found = False

    # Check for any matching combinations
    for count_combination in df.index:
        if set(sorted_combination).issubset(set(count_combination)):
            occurrences = df[count_combination]  # Get the count for the current combination
            total_count += occurrences  # Add the count to the total
            matched_combinations.append((count_combination, occurrences))  # Store the matching combination and count
            print(
                f"The combination {sorted_combination} is part of the Final Four combination {count_combination} and appears {occurrences} times.")
            found = True

    if not found:
        print(f"The combination {sorted_combination} does not exist in any Final Four combination.")
    else:
        print(
            f"The total number of times the combination {sorted_combination} is found: {total_count} times. This is approximately {total_count / 100}% of all final fours")

    # Plotting the matched Final Four combinations if any were found
    if matched_combinations:
        print(f"\nPlotting the matched Final Four combinations:")

        # Convert the matched combinations into a DataFrame
        matched_df = pd.DataFrame(matched_combinations, columns=['Final Four', 'Count'])[:top_x]

        # Creating the plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Count', y='Final Four', data=matched_df, palette='viridis')

        # Adding titles and labels
        plt.title(f'Matched Final Four Combinations for {sorted_combination}')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Final Four Combinations')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()
    else:
        print("No matching combinations were found to plot.")

# Example usage: searching for any combination of 1 to 4 teams and plotting the matched combinations
specific_combination = ['Alabama', 'Connecticut', 'Houston']
check_final_four_combination(specific_combination, final_four_value_counts, 15)

# Plotting the Most common Unique final fours

# Creating Top (x) Final Four combinations
top_20_final_fours = final_four_value_counts.head(20)

# Converting to DF
top_20_df = top_20_final_fours.reset_index()
top_20_df.columns = ['Final Four', 'Count']

# Creating Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Final Four', data=top_20_df, palette='viridis')

# Legend
plt.title('Top 20 Final Four Combinations')
plt.xlabel('Number of Occurrences')
plt.ylabel('Final Four Combinations')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Creating the dictionary for March Madness teams based on their regions

team_region = {
    "Connecticut": "East",
    "Iowa St": "East",
    "Illinois": "East",
    "Auburn": "East",
    "San Diego St": "East",
    "BYU": "East",
    "Washington St": "East",
    "FL Atlantic": "East",
    "Northwestern": "East",
    "Drake": "East",
    "Duquesne": "East",
    "UAB": "East",
    "Yale": "East",
    "Morehead St": "East",
    "S Dakota St": "East",
    "Stetson": "East",

    "North Carolina": "West",
    "Arizona": "West",
    "Baylor": "West",
    "Alabama": "West",
    "St Mary's CA": "West",
    "Clemson": "West",
    "Dayton": "West",
    "Mississippi St": "West",
    "Michigan St": "West",
    "Nevada": "West",
    "New Mexico": "West",
    "Grand Canyon": "West",
    "Col Charleston": "West",
    "Colgate": "West",
    "Long Beach St": "West",
    "Wagner": "West",

    "Houston": "South",
    "Marquette": "South",
    "Kentucky": "South",
    "Duke": "South",
    "Wisconsin": "South",
    "Texas Tech": "South",
    "Florida": "South",
    "Nebraska": "South",
    "Texas A&M": "South",
    "Colorado": "South",
    "NC State": "South",
    "James Madison": "South",
    "Vermont": "South",
    "Oakland": "South",
    "WKU": "South",
    "Longwood": "South",

    "Purdue": "Midwest",
    "Tennessee": "Midwest",
    "Creighton": "Midwest",
    "Kansas": "Midwest",
    "Gonzaga": "Midwest",
    "South Carolina": "Midwest",
    "Texas": "Midwest",
    "Utah St": "Midwest",
    "TCU": "Midwest",
    "Colorado St": "Midwest",
    "Oregon": "Midwest",
    "McNeese St": "Midwest",
    "Samford": "Midwest",
    "Akron": "Midwest",
    "St Peter's": "Midwest",
    "Grambling": "Midwest",
}


# Function to use dictionary above in column creation... Can we just map and get rid of this??
def map_team_to_region(team_name):
    return team_region.get(team_name)

# Starting the process to calculate Region Strength, Adding Region to the df
fragility_calc_df = submission_with_team_info.copy()
fragility_calc_df['Team Region'] = fragility_calc_df['Team'].apply(map_team_to_region)
# fragility_calc_df

# Getting the Total region championships for each region
champions_df = fragility_calc_df[fragility_calc_df['Slot'] == 'R6CH']
region_champ_split = champions_df.value_counts('Team Region')

# Getting the number of championshups each team won
champ_split = champions_df.value_counts('Team')
# champ_split

# Creating Fragility Score
regional_fragility_df = pd.DataFrame(
    {
        'Region': ['East','Midwest','South','West'],
        'Fragility_Score': [(champ_split['Connecticut']/region_champ_split['East']), champ_split['Purdue']/region_champ_split['Midwest'],
                            champ_split['Houston']/region_champ_split['South'], champ_split['Arizona']/region_champ_split['West']]
    }

)
regional_fragility_df = regional_fragility_df.rename(columns = {'Region':'Team Region'})
# regional_fragility_df

# Merging dfs to aid in creation of end Region Strength metric
fragility_calc_df = fragility_calc_df.merge(regional_fragility_df, how = 'outer', on = 'Team Region')
# fragility_calc_df

# Adding the Total Region Championships to df
region_champ_counts = champions_df.value_counts('Team Region').to_dict()
fragility_calc_df['Total Region Championships'] = fragility_calc_df['Team Region'].map(region_champ_counts)/10000
fragility_calc_df

# Running the seed prediction model on this years data to get seed prediction for every team
# Dropping Teams that lose the play in games
kenpom_knn_model_fin_2024 = kenpom_knn_model_2024.copy()
kenpom_knn_model_fin_2024 = kenpom_knn_model_fin_2024.drop(['Boise St.', 'Virginia','Montana St.','Howard'])
kenpom_knn_pred_df = kenpom_knn_model_2024.drop(columns = (['Seed']), axis = 1)

#Introducing the scaling component
scaler = StandardScaler()

# kenpom_knn_pred_df_index = kenpom_knn_pred_df.index
kenpom_knn_pred = scaler.fit_transform(kenpom_knn_pred_df)
seed_preds = knn_best.predict(kenpom_knn_pred)

kenpom_knn_model_2024['Seed_Prediction'] = seed_preds
kenpom_knn_model_2024['Seed_Difference'] = kenpom_knn_model_2024['Seed_Prediction'] - kenpom_knn_model_2024['Seed']
kenpom_knn_model_2024.head(20)

seed_sort = kenpom_knn_model_2024.sort_values(by = 'Seed_Difference', key = lambda x: x.abs(), ascending = False)
# seed_sort.head(20)

# Creating dictionary to toggle between the naming convention of Kenpom and Kaggle datasets
list1 = ['Oregon', 'N.C. State', 'Colorado', 'Stetson', 'Duquesne', 'Kentucky',
         'Akron', 'UAB', 'Oakland', 'Saint Mary\'s', 'Washington St.',
         'New Mexico', 'Kansas', 'Gonzaga', 'Florida', 'Alabama', 'Wisconsin',
         'Michigan St.', 'Clemson', 'Baylor', 'BYU', 'Florida Atlantic', 'Drake',
         'Nevada', 'Northwestern', 'TCU', 'Tennessee', 'Arizona',
         'Mississippi St.', 'Nebraska', 'Illinois', 'Creighton', 'Texas',
         'Saint Peter\'s', 'Western Kentucky', 'Grand Canyon', 'Iowa St.',
         'Colgate', 'South Carolina', 'North Carolina', 'Samford', 'Texas A&M',
         'Long Beach St.', 'South Dakota St.', 'Vermont', 'Longwood',
         'Grambling St.', 'Morehead St.', 'Connecticut', 'Charleston', 'Yale',
         'McNeese St.', 'James Madison', 'Colorado St.', 'Houston', 'Utah St.',
         'Dayton', 'Texas Tech', 'San Diego St.', 'Duke', 'Auburn', 'Marquette',
         'Purdue', 'Wagner']

list2 = ['Connecticut', 'Iowa St', 'Illinois', 'Auburn', 'San Diego St',
         'BYU', 'Drake', 'FL Atlantic', 'Washington St', 'Northwestern',
         'Stetson', 'Yale', 'Duquesne', 'UAB', 'Morehead St', 'S Dakota St',
         'North Carolina', 'Arizona', 'Baylor', 'Alabama', "St Mary's CA",
         'Clemson', 'Nevada', 'Mississippi St', 'Grand Canyon',
         'New Mexico', 'Michigan St', 'Col Charleston', 'Dayton',
         'Long Beach St', 'Colgate', 'Wagner', 'Purdue', 'Tennessee',
         'Creighton', 'Kansas', 'Gonzaga', 'South Carolina', 'Texas',
         'Utah St', 'Colorado St', 'Akron', 'McNeese St', 'TCU', 'Oregon',
         'Samford', "St Peter's", 'Grambling', 'Houston', 'Marquette',
         'Oakland', 'Duke', 'James Madison', 'Texas Tech', 'Florida',
         'Texas A&M', 'Kentucky', 'Wisconsin', 'Nebraska', 'Colorado',
         'WKU', 'NC State', 'Vermont', 'Longwood']

# Create an empty dictionary to store the closest matches
kenpom_kaggle_dict = {}

# Iterate through each team in list1
for team in list1:
    # Find the closest match in list2 using RapidFuzz's process.extractOne
    best_match = process.extractOne(team, list2)

    # Add the match to the dictionary
    kenpom_kaggle_dict[team] = best_match[0]

# Fixing a couple of entries Fuzzy misinterpreted
kenpom_kaggle_dict['Florida Atlantic'] = 'FL Atlantic'
kenpom_kaggle_dict['Western Kentucky'] = 'WKU'

# Getting dfs ready to merge

#Reseting index and creating one Team Column to merge on
seed_sort = seed_sort.reset_index()
seed_sort['Team1'] = seed_sort['Team'].map(kenpom_kaggle_dict)

#Dropping extraneous Team Column and renaming th other
seed_sort = seed_sort.drop(['Team'], axis = 1)
seed_sort = seed_sort.rename(columns = {'Team1':'Team'})

# Grabbing only necessary columns for the Region strength calculation
fragile_calc_df = seed_sort[['Conf','Seed', 'Seed_Prediction', 'Team']]

# Merge
region_strength_df = fragile_calc_df.merge(fragility_calc_df, how = 'inner', on = 'Team' )

#Dropping Duplicates
region_strength_final_df = region_strength_df[['Team', 'Seed_x', 'Seed_Prediction', 'Team Region',
                                            'Fragility_Score','Total Region Championships']].drop_duplicates(subset=['Team'])


# Renaming Seed column
region_strength_final_df = region_strength_final_df.rename(columns = {'Seed_x':'Seed'})


#Groupby to get means of all the variables
Region_seed_prediction_df = region_strength_final_df.groupby('Team Region').mean('Seed_Prediction')

#Calculating Region Strength based on parameters set out
Region_seed_prediction_df['Region Difficulty'] = (1/Region_seed_prediction_df['Fragility_Score'])*Region_seed_prediction_df['Total Region Championships']*((16-Region_seed_prediction_df['Seed_Prediction'])/16)
Region_seed_prediction_df


# Function to predict the chances of a seed going to x round of the tournament
def cinderella(df, seed, slot):
    '''
    df = dataframe of the form from submission function. It needs to have the "Slot" and "Bracket" columns which originate there
         This also needs the "Seed" variable to check the seeds. 'Team' is also included for potential future functionality

    seed = the seed we are checking for round advancement

    slot = the round of the tourmament we are checking the seed has advanced. R1:Round of 32, R2:Sweet Sixteen, R3: Elite 8
           R4: Final Four, R5: Championship Game, R6: Champion
    '''

    count = 0
    # Group by the 'Bracket' column to analyze each tournament separately
    seed_check = df.groupby('Bracket')

    # Loop over each bracket
    for _, group in seed_check:
        # Check if any team reached the specified slot in this bracket
        if group['Slot'].str.contains(slot).any():
            # Find the teams in this bracket that reached the slot
            slot_teams = group[group['Slot'].str.contains(slot)]

            # Check if any of the teams in the specified slot have a seed >= to the given seed
            if (slot_teams['Seed'] >= seed).any():
                count += 1  # Increment the count when the condition is met

    round_dict = {'R1': 'Round of 32', 'R2': 'Sweet Sixteen', 'R3': 'Elite Eight', 'R4': 'Final Four',
                  'R5': 'Championship Game', 'R6': 'Champion'}

    # Assuming 10,000 total brackets, return the fraction of brackets meeting the condition
    print(
        f'There is a {(count / len(seed_check)) * 100}% chance a team of {seed} seed or greater advancing to the {round_dict[slot]}')

# Creating df structure for cinderella function
cinderella_df = submission_with_team_info.copy()
cinderella_df = cinderella_df.rename(columns = {'Seed':'Placement'})
cinderella_df['Seed'] = cinderella_df['Team'].map(team_to_seed)
cinderella(cinderella_df, 10, 'R4')