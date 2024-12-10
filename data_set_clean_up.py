import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
file_path = 'match_data_v5.csv'
dataset = pd.read_csv(file_path)

# Rename columns based on the provided order
column_names = [
    "matchId", "blueTeamControlWardsPlaced", "blueTeamWardsPlaced", "blueTeamTotalKills",
    "blueTeamDragonKills", "blueTeamHeraldKills", "blueTeamTowersDestroyed",
    "blueTeamInhibitorsDestroyed", "blueTeamTurretPlatesDestroyed", "blueTeamFirstBlood",
    "blueTeamMinionsKilled", "blueTeamJungleMinions", "blueTeamTotalGold", "blueTeamXp",
    "blueTeamTotalDamageToChamps", "redTeamControlWardsPlaced", "redTeamWardsPlaced",
    "redTeamTotalKills", "redTeamDragonKills", "redTeamHeraldKills", "redTeamTowersDestroyed",
    "redTeamInhibitorsDestroyed", "redTeamTurretPlatesDestroyed", "redTeamMinionsKilled",
    "redTeamJungleMinions", "redTeamTotalGold", "redTeamXp", "redTeamTotalDamageToChamps",
    "blueWin"
]

# Assign the column names to the dataset
dataset.columns = column_names + ['extra_column']  # Assumes an extra column exists

# Step 1: Remove the last column and matchId
dataset = dataset.iloc[:, :-1]  # Remove the last column
dataset = dataset.drop(columns=['matchId'], errors='ignore')  # Remove matchId column if present

# Step 2: Normalize high-magnitude attributes using Z-score
high_magnitude_attributes = [
    'blueTeamTotalGold', 'redTeamTotalGold',
    'blueTeamXp', 'redTeamXp',
    'blueTeamTotalDamageToChamps', 'redTeamTotalDamageToChamps'
]
scaler = StandardScaler()
dataset[high_magnitude_attributes] = scaler.fit_transform(dataset[high_magnitude_attributes])

# Step 3: Apply Min-Max Scaling for fixed range attributes
fixed_range_attributes = [
    'blueTeamTowersDestroyed', 'redTeamTowersDestroyed',
    'blueTeamInhibitorsDestroyed', 'redTeamInhibitorsDestroyed'
]
min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # Use absolute values in range [0, 1]
dataset[fixed_range_attributes] = min_max_scaler.fit_transform(dataset[fixed_range_attributes])

# Step 4: Cap outliers in wards placed
ward_threshold = 200
dataset['blueTeamControlWardsPlaced'] = dataset['blueTeamControlWardsPlaced'].clip(upper=ward_threshold)
dataset['redTeamControlWardsPlaced'] = dataset['redTeamControlWardsPlaced'].clip(upper=ward_threshold)

# Step 5: Redistribute turret plates and towers due to data errors
def redistribute_plates(row, fraction, blue_col, red_col):
    total_plates = row[blue_col] + row[red_col]
    if total_plates > 0:
        adjustment = total_plates * fraction
        row[blue_col] -= adjustment
        row[red_col] += adjustment
    return row

# Calculate mean values and minion fraction for redistribution
blue_cols = ['blueTeamTurretPlatesDestroyed', 'blueTeamTowersDestroyed']
red_cols = ['redTeamTurretPlatesDestroyed', 'redTeamTowersDestroyed']

for blue_col, red_col in zip(blue_cols, red_cols):
    blue_mean = dataset[blue_col].mean()
    red_mean = dataset[red_col].mean()
    mean_difference = blue_mean - red_mean
    sum_means = blue_mean + red_mean
    minion_fraction = mean_difference / sum_means if sum_means != 0 else 0

    # Apply redistribution to each row
    dataset = dataset.apply(redistribute_plates, axis=1, fraction=minion_fraction, blue_col=blue_col, red_col=red_col)

# Step 6: Add derived attributes (optional)
# Early-game aggression indicator
# dataset['blueAggression'] = (
#     dataset['blueTeamTotalKills'] + 
#     dataset['blueTeamTurretPlatesDestroyed'] + 
#     dataset['blueTeamFirstBlood']
# )
# dataset['redAggression'] = (
#     dataset['redTeamTotalKills'] + 
#     dataset['redTeamTurretPlatesDestroyed'] + 
#     dataset['redTeamFirstBlood']
# )

# Step 7: Save the preprocessed dataset
output_file = 'cleaned_data.csv'
dataset.to_csv(output_file, index=False)

print(f"Data preprocessing complete. Cleaned dataset saved to '{output_file}'.")
