#Import statements

#Standard Impors
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import seaborn as sns

#Model Prep
from sklearn.preprocessing import LabelEncoder, StandardScaler

#setting np random seed
np.random.seed(42)
#https://www.kaggle.com/competitions/march-machine-learning-mania-2024/data
#importing data
#Teams
mteams = pd.read_csv("data/MTeams.csv")
wteams = pd.read_csv('data/WTeams.csv')

#Renaming Team Column for both
mteams = mteams.rename(columns = {'TeamName':'Team'})
wteams = wteams.rename(columns = {'TeamName':'Team'})

#kenpom
kenpom_raw_df = pd.read_csv('data/kenpom_raw.csv')

#Spliting Team to get tournament seed and dropping ranks for all other stats
kenpom_df = kenpom_raw_df.drop(['Rk', 'Unnamed: 6','Unnamed: 8','Unnamed: 10','Unnamed: 12','Unnamed: 14','Unnamed: 16'
                                ,'Unnamed: 18','Unnamed: 20'], axis = 1)

kenpom_df[['Team', 'Seed']] = kenpom_df['Team'].str.extract(r'^(.*?)(\d*)$')
kenpom_df['Team'] = kenpom_df['Team'].str.rstrip()

#The above code was used to fix some issues with the raw file and the rest was fixed in the actual excel file since it was
#easier to do that way. Then it was loaded in the next line

#Re import kenpom after fixing names
kenpom_fin_df = pd.read_csv('data/kenpom_data.csv')

# Not filled observations in Seed column are not recognized as nan so need to converted before the column can be filled and
# finally converted to dtype int
kenpom_fin_df['Seed'] = kenpom_fin_df['Seed'].replace('', np.nan)
kenpom_fin_df['Seed'] = kenpom_fin_df['Seed'].fillna(99)
kenpom_fin_df['Seed'] = kenpom_fin_df['Seed'].astype(int)


# For some of the years in the dataframe range the NIT seed is contained in the Seed column (Only for seeds 1 - 8). The
# Function below are to work around these cases

def limit_seeds(df):
    # Sort the dataframe by Year and Seed
    df = df.sort_values(by=['Year', 'Seed'])

    # Define the special case limits for specific seeds
    # Play-in games can have up to 6 seeds for certain seeds (16, 10, 11, or 12) in some years
    special_seed_limits = {
        16: 6,  # Up to 6 teams allowed for seed 16
        10: 6,  # Only one of 10, 11, or 12 can have 6 teams, others have 4
        11: 6,  # We'll handle these dynamically based on data for each year
        12: 6
    }

    # Function to determine which of the 10, 11, 12 seeds have extra teams in a given year
    def handle_special_cases(df, year):
        if isinstance(year, (list, pd.Series, np.ndarray)):
            raise ValueError("The 'year' parameter should be a single integer or float value.")

        # Calculate counts for seeds 10, 11, and 12
        seed_counts = {
            10: df[(df['Year'].between(year - 0.5, year + 0.5)) & (df['Seed'] == 10)].shape[0],
            11: df[(df['Year'].between(year - 0.5, year + 0.5)) & (df['Seed'] == 11)].shape[0],
            12: df[(df['Year'].between(year - 0.5, year + 0.5)) & (df['Seed'] == 12)].shape[0]
        }

        # List comprehension to include all seeds with counts > 4
        extra_teams_seeds = [seed for seed, count in seed_counts.items() if count > 4]

        # Return list of seeds with extra teams, or None if none exceed 4
        return extra_teams_seeds if extra_teams_seeds else []

    # Group by Year and Seed, then count occurrences of each seed
    df['Seed_Count'] = df.groupby(['Year', 'Seed']).cumcount() + 1

    # Iterate through each year to apply limits
    for year in df['Year'].unique():

        # Handle seed 16 separately (always 6 teams in play-in cases)
        mask_16 = (df['Year'].between(year - 0.5, year + 0.5)) & (df['Seed'] == 16)
        df.loc[mask_16 & (df['Seed_Count'] > special_seed_limits[16]), 'Seed'] = 99

        # Determine which seed (10, 11, or 12) has the extra teams
        extra_seeds = handle_special_cases(df, year)

        # Apply the special limit for seeds with extra teams
        if extra_seeds is not None:
            for seed in extra_seeds:
                mask_extra = (df['Year'] == year) & (df['Seed'] == seed)
                df.loc[mask_extra & (df['Seed_Count'] > special_seed_limits[seed]), 'Seed'] = 99
        # Apply the default limit of 4 for other seeds
        for seed in range(1, 16):
            if seed not in extra_seeds:
                mask_other = (df['Year'] == year) & (df['Seed'] == seed)
                df.loc[mask_other & (df['Seed_Count'] > 4), 'Seed'] = 99

    # Drop the helper Seed_Count column
    df.drop('Seed_Count', axis=1, inplace=True)

    return df

# Apply fuunction and sanity check
kenpom_fin_df = limit_seeds(kenpom_fin_df)
seed_check = kenpom_fin_df.value_counts('Seed')
# seed_check

#merge with MTeams to add Team ID to kenpom
kenpom_fin_df = kenpom_fin_df.merge(mteams, how = 'inner', on = 'Team')
kenpom_fin_df = kenpom_fin_df.rename(columns = {'AdjEM.1':'AdjEM_SOS','AdjEM.2':'AdjEM_NCSOS', 'Year':'Season'})

# Importing Seasons, Tournament seeds and results
# Not all data was used, further models could use more data or already used data different, this comment is being made once
# the code as a whole was completed, at the start everything that seemed halfway useable was loaded

# Seasons
mseasons_df = pd.read_csv('data/MSeasons.csv')
wseasons_df = pd.read_csv('data/WSeasons.csv')

# Seeds
mseeds_df = pd.read_csv('data/MNCAATourneySeeds.csv')
wseeds_df = pd.read_csv('data/WNCAATourneySeeds.csv')

# Regular season results - compact
mregresultscomp_df = pd.read_csv('data/MRegularSeasonCompactResults.csv')
wregresultscomp_df = pd.read_csv('data/WRegularSeasonCompactResults.csv')

# Tournament Results - compact
mmadresultscomp_df = pd.read_csv('data/MNCAATourneyCompactResults.csv')
wmadresultscomp_df = pd.read_csv('data/WNCAATourneyCompactResults.csv')

# Regular season results - detailed
mregresultsdet_df = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
wregresultsdet_df = pd.read_csv('data/WRegularSeasonDetailedResults.csv')

# Tournament Results - detailed
mmadresultsdet_df = pd.read_csv('data/MNCAATourneyDetailedResults.csv')
wmadresultsdet_df = pd.read_csv('data/WNCAATourneyDetailedResults.csv')

# Coaches
mcoaches_df = pd.read_csv('data/MTeamCoaches.csv')

# Conferences
conf_df = pd.read_csv('data/Conferences.csv')
mconf_df = pd.read_csv('data/MTeamConferences.csv')
wconf_df = pd.read_csv('data/WTeamConferences.csv')

# Conference Tournament Results and Non March Madness tournament teams + results
mconftourn_df = pd.read_csv('data/MConferenceTourneyGames.csv')
msectournteams_df = pd.read_csv('data/MSecondaryTourneyTeams.csv')
msectournres_df = pd.read_csv('data/MSecondaryTourneyCompactResults.csv')

# Tournament Slots
mtournslot_df = pd.read_csv('data/MNCAATourneySlots.csv')
wtournslot_df = pd.read_csv('data/WNCAATourneySlots.csv')
mseedslot_df = pd.read_csv('data/MNCAATourneySeedRoundSlots.csv')

#Kenpom data only goes to 2003 so we are droping anything before

# List of all DataFrames for below for loop
dataframes = [mseasons_df, wseasons_df, mseeds_df, wseeds_df,
              mregresultscomp_df, wregresultscomp_df, mmadresultscomp_df,
              wmadresultscomp_df, mregresultsdet_df, wregresultsdet_df,
              mmadresultsdet_df, wmadresultsdet_df, mcoaches_df, conf_df,
              mconf_df, wconf_df, mconftourn_df, msectournteams_df,
              msectournres_df, mtournslot_df, mseedslot_df]

# Removing years not in kenpom data
for df in dataframes:
    # Check if 'Year' or 'Season' column exists
    if 'Year' in df.columns:
        # Drop rows where 'Year' is before 2003
        df.drop(df[(df['Year'] < 2003)].index, inplace=True)
    elif 'Season' in df.columns:
        # Drop rows where 'Season' is before 2003
        df.drop(df[(df['Season'] < 2003)].index, inplace=True)

# Copying winning Team to new column
mregresultscomp_df.reset_index(inplace=True)
mregresultscomp_df['Winner'] = mregresultscomp_df['WTeamID']

# Seperating the TeamIDs for Merge
# Need to merge kenpom data for each team. In order to accomplish the oringal df is being split on TeamID, merged by team
# with kenpom so that each on row will be a game with each team and their respective kenpom stats
wgames_df = mregresultscomp_df[['index','Season','DayNum', 'WTeamID','WScore','WLoc', 'NumOT','Winner']]
lgames_df = mregresultscomp_df[['index','Season','DayNum', 'LTeamID','LScore','WLoc', 'NumOT']]

# Renaming TeamID Columns to Team ID, will be renamed to TeamID_A and TeamID_B on final merge
wgames_df = wgames_df.rename(columns = {'WTeamID':'TeamID'})
lgames_df = lgames_df.rename(columns = {'LTeamID':'TeamID'})

# Merging with kenpom
wgames_df = wgames_df.merge(kenpom_fin_df, on =['TeamID','Season'])
lgames_df = lgames_df.merge(kenpom_fin_df, on =['TeamID','Season'])

# Recreating for Tournament games
# Copying winning Team to new column and switching everything to
mmadresultsdet_df['Winner'] = mmadresultsdet_df['WTeamID']

# Seperating the TeamIDs for Merge
wgamestourn_df = mmadresultsdet_df[['Season','DayNum', 'WTeamID','WScore','WLoc', 'NumOT','Winner']]
lgamestourn_df = mmadresultsdet_df[['Season','DayNum', 'LTeamID','LScore','WLoc', 'NumOT']]

# Renaming TeamID Columns to Team ID, will be renamed to TeamID_A and TeamID_B on final merge
wgamestourn_df = wgamestourn_df.rename(columns = {'WTeamID':'TeamID'})
lgamestourn_df = lgamestourn_df.rename(columns = {'LTeamID':'TeamID'})

# Merging with kenpom
wgamestourn_df = wgamestourn_df.merge(kenpom_fin_df, on =['TeamID','Season'])
lgamestourn_df = lgamestourn_df.merge(kenpom_fin_df, on =['TeamID','Season'])


# Merging everthing together
games_df = wgames_df.merge(lgames_df, on = 'index' ,suffixes = ('_A', '_B'))
gamestourn_df = wgamestourn_df.merge(lgamestourn_df, left_index = True, right_index = True ,suffixes = ('_A', '_B'))

# Final df form for analysis
allgames_df = pd.concat([games_df, gamestourn_df], axis = 0)

# Prepping Everything for analysis
allgames_df = allgames_df.drop(columns = ['index','Season_B', 'DayNum_B','WLoc_B','NumOT_B'])#'W-L_A','W-L_B',
allgames_df = allgames_df.rename(columns = {'Season_A':'Season', 'DayNum_A':'DayNum','WLoc_A':'WLoc','NumOT_A':'NumOT'})

# Split the W-L column into two columns: W and L
allgames_df[['W_A', 'L_A']] = allgames_df['W-L_A'].str.split('-', expand=True)
allgames_df[['W_B', 'L_B']] = allgames_df['W-L_B'].str.split('-', expand=True)

# Convert both columns to numeric
allgames_df['W_A'] = pd.to_numeric(allgames_df['W_A'])
allgames_df['L_A'] = pd.to_numeric(allgames_df['L_A'])

# Convert both columns to numeric
allgames_df['W_B'] = pd.to_numeric(allgames_df['W_B'])
allgames_df['L_B'] = pd.to_numeric(allgames_df['L_B'])

# Dropping more columns
allgames_df = allgames_df.drop(columns=['Team_A', 'Team_B', 'WLoc', 'WScore', 'LScore', 'NumOT', 'DayNum', 'Winner'])

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
allgames_df['Conf_A'] = allgames_df['Conf_A'].map(conf_mapping)
allgames_df['Conf_B'] = allgames_df['Conf_B'].map(conf_mapping)

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
allgames_df['Conf_Tier_A'] = allgames_df['Conf_A'].apply(map_tier)
allgames_df['Conf_Tier_B'] = allgames_df['Conf_B'].apply(map_tier)

# Create LabelEncoder object (for conferences)
label_encoder = LabelEncoder()

# Fit and transform Conf_A and Conf_B columns to integers
allgames_df['Conf_A'] = label_encoder.fit_transform(allgames_df['Conf_A'])
allgames_df['Conf_B'] = label_encoder.fit_transform(allgames_df['Conf_B'])

# Function to determine random team ID
def random_id(row):
    return np.random.choice([row['TeamID_A'], row['TeamID_B']])

#Create the new column 'RandID' by applying random_id function
allgames_df['RandID'] = allgames_df.apply(random_id, axis=1)

#Creating variable to be dependent variable in model
allgames_df['WinnerID'] = allgames_df['TeamID_A']

#Saving final df so it can be visualized easier in case of debugging
allgames_df.to_csv('data/_allgamescheck.csv', index = False)


# Function to make sure that the winning team is not always listed first in the dataframe
def stat_swap(df):
    # Empty list to store the result rows
    selected_rows = []

    # Static columns that should always be present
    static_columns = ['Season', 'RandID', 'WinnerID']

    # Get the columns ending in _A and _B
    columns_A = [col for col in df.columns if col.endswith('_A')]
    columns_B = [col for col in df.columns if col.endswith('_B')]

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # If RandID == TeamID_A, we keep the values as is
        if row['RandID'] == row['TeamID_A']:
            selected_data = row.to_dict()

        # If RandID == TeamID_B, we swap values between _A and _B columns
        else:
            selected_data = row.to_dict()  # Start by copying the row data

            # Swap the values of _A and _B columns
            for col_A, col_B in zip(columns_A, columns_B):
                selected_data[col_A], selected_data[col_B] = selected_data[col_B], selected_data[col_A]

        # Add static columns to the result
        result_row = {col: selected_data[col] for col in static_columns + columns_A + columns_B}
        selected_rows.append(result_row)

    # Convert the result list into a DataFrame
    selected_df = pd.DataFrame(selected_rows)

    return selected_df

# Applying stat swap
allgames_df2 = stat_swap(allgames_df)

# Creating variable to be dependent variable in model
allgames_df2['Winner'] = (allgames_df2['WinnerID'] == allgames_df2['RandID']).astype(int)

# Sanity check to make sure winner is being correctly randomally assigned.
# Current process is to randomly chose a TeamID in the matchup data and compare it to the WinnerID if the
# two are the same Winner == 1 else Winner == 0
# sum(allgames_df2['Winner'])/len(allgames_df2)

# Dropping both TeamIDs so that model does not see that TeamID_A is always the winner
allgames_df2 = allgames_df2.drop(columns = (['W-L_A','W-L_B','RandID','WinnerID','W_A','L_A','W_B','L_B']))#['TeamID_A', 'TeamID_B',]

# Filling Seed variable for teams that did not make the the tournament, might need to add logic for NIT
allgames_df2['Seed_A'] = allgames_df2['Seed_A'].fillna(99)
allgames_df2['Seed_B'] = allgames_df2['Seed_B'].fillna(99)

# Saving final df so it can be visualized easier in case of debugging
allgames_df2.to_csv('data/_allgames2.csv', index = False)

# Dropping columns which proved to be extraneous in the model to get a more streamlined, less complex outcome
allgames_df2 = allgames_df2.drop(columns = (['Luck_A','Luck_B','AdjEM_A','AdjEM_B','FirstD1Season_B', 'LastD1Season_B','FirstD1Season_A', 'LastD1Season_A']))
