import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
#imporstatments from nccabwinsmodel
#imporing model related variables
from ncaabwins_logreg import nccabwins, X

#importing dataframes
from ncaabwinsmodel import mteams, mseeds_df, mtournslot_df,kenpom_fin_df

#Imporing dictionaries
from ncaabwinsmodel import conf_mapping

#importing functions
from ncaabwinsmodel import random_id, stat_swap

#Load necessary data (team IDs, features, etc.)
teams_2024_raw = pd.read_csv('data/2024_tourney_seeds.csv')

#Creating a TeamID/Team Name dictionary
team_dict = dict(zip(mteams['TeamID'], mteams['Team']))

#Grabbing just the Mens teams (this is a model and sim for the mens tournament)
teams_2024_dfm = teams_2024_raw[teams_2024_raw['Tournament'] == 'M']


#Generate all possible pairs of teams
teams_pairs = []
for team1_id in teams_2024_dfm['TeamID']:
    for team2_id in teams_2024_dfm['TeamID']:
        if team1_id != team2_id:  # Exclude same team pairs
            teams_pairs.append((team1_id, team2_id))


#Get the index of the 'Season' column
season_index = kenpom_fin_df.columns.get_loc('Season')

#Move the 'Season' column to the first position
cols = list(kenpom_fin_df.columns)
cols.insert(0, cols.pop(season_index))
kenpom_fin_df = kenpom_fin_df[cols]

kenpom_fin_df = kenpom_fin_df.drop(columns = ['Team','W-L'])
kenpom_fin_df['Conf'] = kenpom_fin_df['Conf'].map(conf_mapping)
label_encoder = LabelEncoder()
kenpom_fin_df['Conf'] = label_encoder.fit_transform(kenpom_fin_df['Conf'])
def prep_data_pred(team1_id, team2_id):
    #Function takes in two teams and returns a one row dataframe which is identical to the dataframe which the random
    #forest model from nccabwinsmodel is trained on. To see its column names and order print(X.columns)

    #team1_id: id associated with team1
    #team2_id: id associated with team1

    #Data to add to pairs

    #Selected only data from 2024 season
    kenpom_2024_df = kenpom_fin_df[kenpom_fin_df['Season']==2024]

    #Getting the kenpom data for both Team1 and Team2 and renaming the columns to be the same as from the model
    row1 = kenpom_2024_df[kenpom_2024_df['TeamID'] == team1_id]
    row1.columns = [col + "_A" for col in row1.columns]
    row1 = row1.rename(columns = {'Season_A':'Season'})
    row2 = kenpom_2024_df[kenpom_2024_df['TeamID'] == team2_id]
    row2.columns = [col + "_B" for col in row2.columns]
    #Dropping Season_B, not needed
    row2 = row2.drop(columns = ['Season_B'])

    #Combining the rows into row/df
    row1, row2 = row1.reset_index(drop = True), row2.reset_index(drop = True)
    stats_df = pd.concat([row1, row2],axis =1)

    #Creating RandID just as it appears in model
    stats_df['RandID'] = stats_df.apply(random_id, axis=1)
    stats_df =stat_swap(stats_df)

    #Dropping TeamA and TeamB because the model would understand that TeamA is always the winner
    stats_df =stats_df.drop(columns = ['TeamID_A', 'TeamID_B'])

    #Reindex the DataFrame with the desired column order
    stats_df = stats_df.reindex(columns=X.columns)
    return stats_df

#Make predictions for each team pair
predictions = []
for team1_id, team2_id in teams_pairs:

    #Prepare features for prediction
    features = prep_data_pred(team1_id, team2_id)

    #Make prediction using the model
    prediction = nccabwins.predict_proba(features)
    predictions.append((team1_id, team2_id, prediction))

#Format predictions into DataFrame
predictions_df = pd.DataFrame(predictions, columns=['TeamID1', 'TeamID2', 'Prediction'])

#Save predictions to a CSV file
predictions_df.to_csv('data/_tournament_predictions.csv', index=False)

def prep_data_sim(seeds, preds):
    # Function preparing the data for the simulation
    seed_dict = seeds.set_index('Seed')['TeamID'].to_dict()
    inverted_seed_dict = {value: key for key, value in seed_dict.items()}
    probs_dict = {}

    for team1, team2, probs in zip(preds['TeamID1'], preds['TeamID2'], preds['Prediction']):
        #print(preds['Prediction'])
        probs_dict.setdefault(team1, {})[team2] = 1-probs[0]
        probs_dict.setdefault(team2, {})[team1] = probs[0]

    return seed_dict, inverted_seed_dict, probs_dict

preds = predictions_df

#Getting the seeds and tournament slots for the mens tournament in 2024
mseeds_df2024 = mseeds_df[mseeds_df['Season'] == 2024]
mtournslot_df2024 = mtournslot_df[mtournslot_df['Season'] == 2024]

#Getting the data from prep_data_sims to be examined to better under stand process
seed_dict, inverted_seed_dict, probs_dict = prep_data_sim(mseeds_df2024, preds)

#Saving used data to dfs and the to csvs to better understand the process of what is happening, this is not necessary
#for the code to run
probs_df = pd.DataFrame.from_dict(probs_dict, orient='index')
probs_df.to_csv('data/_probs_dict.csv')
seed_dict = pd.DataFrame.from_dict(seed_dict, orient='index')
seed_dict.to_csv('data/_seed_dict.csv')
inverted_seed_dict = pd.DataFrame.from_dict(inverted_seed_dict, orient='index')
inverted_seed_dict.to_csv('data/_inverted_seed_dict.csv')
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




result_m = run_simulation(brackets=10000, seeds=mseeds_df2024, preds=preds, round_slots=mtournslot_df2024, sim=True)
#result_m['Tournament'] = 'M'
submission = result_m
submission.reset_index(inplace=True, drop=True)
submission.index.names = ['RowId']
submission = submission.drop(columns = 'Tournament')
submission =submission.rename(columns = {'Team':'Seed'})

#Saving submission file to examine what is happening
submission.to_csv('data/_submission.csv', index=False)

#Expanding on submission file so it is easier to read for humans
result_analysis_df = submission.merge(mseeds_df2024, on = 'Seed', how = 'inner')

#Dropping season and adding team ID for analysis on results
submission_with_team_info = result_analysis_df.drop(columns = 'Season')
submission_with_team_info = result_analysis_df.merge(mteams, on = 'TeamID', how = 'inner')

#Saving new submmi
result_analysis_df.to_csv('data/_result_analysis.csv', index=False)



#Calculating how many times each team makes it to X round
#Initialize dictionaries to store counts for each round
sweet_sixteen_counts = {}
elite_eight_counts = {}
final_four_counts = {}
championship_counts = {}

# Iterate over each row in the DataFrame
for index, row in submission_with_team_info.iterrows():
    #Get the team in this row
    team = row['Team']

    #Get the round of this match
    round_name = row['Slot'].split('_')[0]

    #Increment the count for the respective round for this team
    if 'R2' in round_name:
        sweet_sixteen_counts[team] = sweet_sixteen_counts.get(team, 0) + 1
    elif 'R3' in round_name:
        elite_eight_counts[team] = elite_eight_counts.get(team, 0) + 1
    elif 'R4' in round_name:
        final_four_counts[team] = final_four_counts.get(team, 0) + 1
    elif 'R6' in round_name:
        championship_counts[team] = championship_counts.get(team, 0) + 1

#Create DataFrames from the dictionaries
sweet_sixteen_df = pd.DataFrame(list(sweet_sixteen_counts.items()), columns=['Team', 'SweetSixteenCount'])
elite_eight_df = pd.DataFrame(list(elite_eight_counts.items()), columns=['Team', 'EliteEightCount'])
final_four_df = pd.DataFrame(list(final_four_counts.items()), columns=['Team', 'FinalFourCount'])
championship_df = pd.DataFrame(list(championship_counts.items()), columns=['Team', 'ChampionshipCount'])

#Merge all DataFrames on 'Team'
round_counts_df = pd.merge(sweet_sixteen_df, elite_eight_df, on='Team', how='outer')
round_counts_df = pd.merge(round_counts_df, final_four_df, on='Team', how='outer')
round_counts_df = pd.merge(round_counts_df, championship_df, on='Team', how='outer')

#Fill NaN values with 0
round_counts_df = round_counts_df.fillna(0)
print(mteams.columns, round_counts_df.columns)
round_counts_df = round_counts_df.merge(mteams, on = 'Team', how ='inner')
round_counts_df = round_counts_df.drop(columns = ['FirstD1Season','LastD1Season'])
#Save round_counts_df to a CSV file
round_counts_df.to_csv('data/_round_counts.csv', index=False)

