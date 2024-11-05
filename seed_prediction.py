# Import Statements

#General Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Model Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Reading in kenpom data
kenpom_raw_df = pd.read_csv('data/kenpom_raw.csv')

# Spliting Team to get tournament seed and dropping ranks for all other stats
kenpom_df = kenpom_raw_df.drop(['Rk', 'Unnamed: 6','Unnamed: 8','Unnamed: 10','Unnamed: 12','Unnamed: 14','Unnamed: 16'
                                ,'Unnamed: 18','Unnamed: 20'], axis = 1)
kenpom_df[['Team', 'Seed']] = kenpom_df['Team'].str.extract(r'^(.*?)(\d*)\*?$')
kenpom_df['Team'] = kenpom_df['Team'].str.rstrip()

# Creating Kenpom seed which is sorting on NetRtg (efficency on both sides of the ball) and giving every four teams the next numerical seed
# The first four teams get 1 the next four get 2, etc etc until the 64th team gets 16
def assign_kenpom_seed(df):
    # Sort by 'NetRtg' within each year and assign ranking
    df = df.sort_values(by='AdjEM', ascending=False).reset_index(drop=True)

    # Assign seeds 1 to 16 for top 64 teams, remaining get seed 99
    df['kenpom_seed'] = np.where(df.index < 64, (df.index // 4) + 1, 99)

    return df

# Group by 'year' and apply the seed assignment function
kenpom_df = kenpom_df.groupby('Year', group_keys=False).apply(assign_kenpom_seed)

#This works really well for the top end which is mostly what we care about but is a disaster at the lower end since
# a bunch of smaller schools make the tournament and take the 13 - 16 seeds. There are ways to fix this (Described below)
# but for the moment this is more than sufficient since this is not going to be used on the model. And is more of a comparison tool
# for the upper end on how the model does.

# This can fixed by including this for the top 50 teams then looking at teams in the tier 3 of conferences and randomizing the
# First few teams in each mid major conference a seed between 13-16. If these conferences have a team in the top 50 then we can
# assume they are the conference representatitve and ignore this process.

# Not filled observations in Seed column are not recognized as nan so need to converted before the column can be filled and
# finally converted to dtype int
kenpom_df['Seed'] = kenpom_df['Seed'].replace('', np.nan)
kenpom_df['Seed'] = kenpom_df['Seed'].fillna(99)
kenpom_df['Seed'] = kenpom_df['Seed'].astype(int)

# Creating variable for see difference
kenpom_df['Seed_Difference'] = kenpom_df['Seed'].astype(int) - kenpom_df['kenpom_seed'].astype(int)
kenpom_df['Seed_Difference'] = kenpom_df['Seed_Difference'].astype(int)


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

# Apply limit seeds to get rid of NIT random seeds interfering
kenpom_df = limit_seeds(kenpom_df)

# Creating variable for seed difference
kenpom_df['Seed_Difference'] = kenpom_df['Seed'].astype(int) - kenpom_df['kenpom_seed'].astype(int)
kenpom_df['Seed_Difference'] = kenpom_df['Seed_Difference'].astype(int)

# Create snub df to see which teams Kenpom said should have been in in previous years
kenpom_snub_df = kenpom_df.sort_values(by='Seed_Difference', key=lambda x: x.abs(), ascending=False)
# kenpom_snub_df.head(20)
# kenpom_df

kenpom_misseed_df = kenpom_snub_df[kenpom_snub_df['Seed_Difference'].abs() < 20]
# kenpom_misseed_df.head(25)

# From the above we can see that the Kenpom seeding is probably a more accurate reflection of a teams strength below are the
# outcomes of all the listed teams
# Vanderbilt 2008: Smoked in first round by 13 seed Siena
# Davidson 2008: Made Elite 8 and lost by 2 to eventual champion Kansas
# Oregon St. 2016: Lost to VCU in first game
# Dayton 2016: lost to Syracuse in first game
# Wichita St. 2016: Won play in and second round game bfore losiing in round of 32 to 3 seed Miami
# DePaul 2004 beat Dayton in first game and lost in second round to Uconn
# Washington 2004: Lost to UAB in first game
# Wichita St 2017 Lost to Kentucky (2 seed) in second game
# Missouri 2023: beat UT St. in first game but the lost to 15 seed Princeton in next game
# Utah St. 2005: Lost to Arizona (3 seed) in first game
# Charlotte 2005: Lost to NC State in first game
# Tennessee 2014: Made Sweet 16 but did not have to beat 3 seed Duke since the lost to Mercer in their first game
# UMass 2014: Played Tennessee (from above) in first round and lost
# Virginia 2007: Lost to Tennessee in second round 4 v 5 seed
# UCLA 2021: Lost to Gonzaga in Final Four
# Belmont 2012: Lost to Georgetown (3 seed) in first round
# Minnesota 2012: Not in Tourney??
# Richmond 2010: Lost in first game to Saint Mary's
# Wisconsin 2022: Lost to 11 seed Iowa St. in second game
# South Carolina 2024: Lost to Oregon (11 Seed) in first game
# Belmont 2011: Lost to Wisconsin (4 seed) in first game
# Colorado 2016: Lost to Uconn in 8,9 mathcup
# Tennessee 2011: Smoked by Michigan in 8,9 matchup


# Prepping dataframe for algorithm

# Define conference tiers definition based on 2024 seeding results
conference_tiers = {
    'SEC': 1, 'B12': 1, 'B10': 1, 'ACC': 1, 'BE': 1, 'P12': 1, 'MWC': 1, 'P10':1,
    'A10': 2, 'AMER': 2, 'MVC': 2, 'WCC': 2
}

# Assign Tier 3 to the rest of the conferences
all_conferences = set(kenpom_df['Conf'].unique())
tier_three_conferences = all_conferences - set(conference_tiers.keys())
conference_tiers.update({conf: 3 for conf in tier_three_conferences})

# Create a function to map conferences to tiers
def map_tier(conf):
    return conference_tiers.get(conf, 3)  # Default to tier 3 if not found

# Apply conference tier mapping to Conf_A and Conf_B columns
kenpom_df['Conf_Tier'] = kenpom_df['Conf'].apply(map_tier)

# Create LabelEncoder object (for conferences)
label_encoder = LabelEncoder()

# Fit and transform Conf column to integers
kenpom_df['Conf'] = label_encoder.fit_transform(kenpom_df['Conf'])

# Renaming a few columns
kenpom_df = kenpom_df.rename(columns = {'AdjEM':'NetRtg','AdjEM.1':'SOS_NetRtg','AdjEM.2':'NCSOS_NetRtg'})

# Split the W-L column into two columns: W and L
kenpom_df[['W', 'L']] = kenpom_df['W-L'].str.split('-', expand=True)

# Convert both columns to numeric
kenpom_df['W'] = pd.to_numeric(kenpom_df['W'])
kenpom_df['L'] = pd.to_numeric(kenpom_df['L'])

# Dropping non-tournament teams
kenpom_tourney = kenpom_df[kenpom_df['Seed']<20]
kenpom_tourney = kenpom_tourney.sample(frac = 1).reset_index(drop = True)
# kenpom_tourney

# Getting df ready for model and the 2024 ready for implementation in Sims
kenpom_knn_model_df = kenpom_tourney.drop(['W-L','kenpom_seed','Seed_Difference'], axis = 1)
kenpom_knn_model_df = kenpom_knn_model_df.set_index('Team')
kenpom_knn_model_2024 = kenpom_knn_model_df[kenpom_knn_model_df['Year'] == 2024]
kenpom_knn_model_df_fin = kenpom_knn_model_df[kenpom_knn_model_df['Year'] != 2024]
# kenpom_knn_model_2024

# Define model variables
X = kenpom_knn_model_df_fin.drop('Seed', axis = 1)  # Example features
y = kenpom_knn_model_df_fin['Seed'].astype(int)  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter Grid
param_grid = {'n_neighbors': np.arange(1, 31)}

# Initial Classifier
knn = KNeighborsClassifier()

# Grid Search
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X_train, y_train)

# Get the best K
best_k = knn_gscv.best_params_['n_neighbors']
print(f"Best value for k: {best_k}")

# Train the model with the best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Make predictions
y_pred = knn_best.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy for different values of k
k_values = np.arange(1, 31)
train_scores = []
test_scores = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    train_scores.append(knn_k.score(X_train, y_train))
    test_scores.append(knn_k.score(X_test, y_test))

plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, label='Train Accuracy')
plt.plot(k_values, test_scores, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy for Different k Values')
plt.legend()
plt.grid()
plt.show()
