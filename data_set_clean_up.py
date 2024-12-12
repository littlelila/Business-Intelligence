import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
file_path = 'data_sets/match_data_v5.csv'
dataset = pd.read_csv(file_path)

# Rename columns based on the provided order
column_names = [
    "matchId", 
    "blueTeamControlWardsPlaced", 
    "blueTeamWardsPlaced", 
    "blueTeamTotalKills",
    "blueTeamDragonKills", 
    "blueTeamHeraldKills", 
    "blueTeamTowersDestroyed",
    "blueTeamInhibitorsDestroyed", 
    "blueTeamTurretPlatesDestroyed", 
    "blueTeamFirstBlood",
    "blueTeamMinionsKilled", 
    "blueTeamJungleMinions", 
    "blueTeamTotalGold", 
    "blueTeamXp",
    "blueTeamTotalDamageToChamps", 
    "redTeamControlWardsPlaced", 
    "redTeamWardsPlaced",
    "redTeamTotalKills", 
    "redTeamDragonKills", 
    "redTeamHeraldKills", 
    "redTeamTowersDestroyed",
    "redTeamInhibitorsDestroyed", 
    "redTeamTurretPlatesDestroyed", 
    "redTeamMinionsKilled",
    "redTeamJungleMinions", 
    "redTeamTotalGold", 
    "redTeamXp", 
    "redTeamTotalDamageToChamps",
    "blueWin"
]

# Assign the column names to the dataset
dataset.columns = column_names + ['extra_column']  # Assumes an extra column exists

# Step 1: Remove the last column and matchId
dataset = dataset.iloc[:, :-1]  # Remove the last column
dataset = dataset.drop(columns=['matchId'], errors='ignore')  # Remove matchId column if present

# # Step 2: Normalize high-magnitude attributes using Z-score
# high_magnitude_attributes = [
#     'blueTeamTotalGold', 'redTeamTotalGold',
#     'blueTeamXp', 'redTeamXp',
#     'blueTeamTotalDamageToChamps', 'redTeamTotalDamageToChamps'
# ]
# scaler = StandardScaler()
# dataset[high_magnitude_attributes] = scaler.fit_transform(dataset[high_magnitude_attributes])

# # Step 3: Apply Min-Max Scaling for fixed range attributes
# fixed_range_attributes = [
#     'blueTeamTowersDestroyed', 'redTeamTowersDestroyed',
#     'blueTeamInhibitorsDestroyed', 'redTeamInhibitorsDestroyed'
# ]
# min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # Use absolute values in range [0, 1]
# dataset[fixed_range_attributes] = min_max_scaler.fit_transform(dataset[fixed_range_attributes])

# # Step 4: Cap outliers in wards placed (convert to int after clipping)
# ward_threshold = 200
# dataset['blueTeamWardsPlaced'] = dataset['blueTeamWardsPlaced'].clip(upper=ward_threshold).astype(int)
# dataset['redTeamWardsPlaced'] = dataset['redTeamWardsPlaced'].clip(upper=ward_threshold).astype(int)

# Step 5: Redistribute turret plates and towers due to data errors (ensure redistribution remains integer)
def redistribute_plates(row, fraction, blue_col, red_col):
    total_plates = row[blue_col] + row[red_col]
    if total_plates > 0:
        adjustment = int(total_plates * fraction)  # Convert adjustment to integer
        row[blue_col] = int(row[blue_col] - adjustment / 2)
        row[red_col] = int(row[red_col] + adjustment / 2)
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

# Step 6: Add differences between attributes and ensure they are integers
dataset['GoldDifference'] = (dataset['blueTeamTotalGold'] - dataset['redTeamTotalGold']).astype(int)
dataset['XpDifference'] = (dataset['blueTeamXp'] - dataset['redTeamXp']).astype(int)
dataset['DamageDifference'] = (dataset['blueTeamTotalDamageToChamps'] - dataset['redTeamTotalDamageToChamps']).astype(int)
dataset['KillDifference'] = (dataset['blueTeamTotalKills'] - dataset['redTeamTotalKills']).astype(int)
dataset['DragonKillDifference'] = (dataset['blueTeamDragonKills'] - dataset['redTeamDragonKills']).astype(int)
dataset['HeraldKillDifference'] = (dataset['blueTeamHeraldKills'] - dataset['redTeamHeraldKills']).astype(int)
dataset['TowerDifference'] = (dataset['blueTeamTowersDestroyed'] - dataset['redTeamTowersDestroyed']).astype(int)
dataset['InhibitorDifference'] = (dataset['blueTeamInhibitorsDestroyed'] - dataset['redTeamInhibitorsDestroyed']).astype(int)
dataset['WardDifference'] = (dataset['blueTeamWardsPlaced'] - dataset['redTeamWardsPlaced']).astype(int)
dataset['ControlWardDifference'] = (dataset['blueTeamControlWardsPlaced'] - dataset['redTeamControlWardsPlaced']).astype(int)
dataset['TurretPlatesDifference'] = (dataset['blueTeamTurretPlatesDestroyed'] - dataset['redTeamTurretPlatesDestroyed']).astype(int)
dataset['MinionsDifference'] = (dataset['blueTeamMinionsKilled'] - dataset['redTeamMinionsKilled']).astype(int)
dataset['JungleMinionsDifference'] = (dataset['blueTeamJungleMinions'] - dataset['redTeamJungleMinions']).astype(int)

# Step 7: Drop columns not needed for the final data
columns_to_remove = [
    "blueTeamControlWardsPlaced", 
    "blueTeamWardsPlaced", 
    "blueTeamTotalKills",
    "blueTeamDragonKills", 
    "blueTeamHeraldKills", 
    "blueTeamTowersDestroyed",
    "blueTeamInhibitorsDestroyed", 
    "blueTeamTurretPlatesDestroyed", 
    # "blueTeamFirstBlood",
    "blueTeamMinionsKilled", 
    "blueTeamJungleMinions", 
    "blueTeamTotalGold", 
    "blueTeamXp",
    "blueTeamTotalDamageToChamps", 
    "redTeamControlWardsPlaced", 
    "redTeamWardsPlaced",
    "redTeamTotalKills", 
    "redTeamDragonKills", 
    "redTeamHeraldKills", 
    "redTeamTowersDestroyed",
    "redTeamInhibitorsDestroyed", 
    "redTeamTurretPlatesDestroyed", 
    "redTeamMinionsKilled",
    "redTeamJungleMinions", 
    "redTeamTotalGold", 
    "redTeamXp", 
    "redTeamTotalDamageToChamps",
    # "blueWin"
]
dataset = dataset.drop(columns=columns_to_remove)

# Ensure all remaining columns are integers (optional final step)
dataset = dataset.round().astype(int)

# Step 8: Save the preprocessed dataset
output_file = 'data_sets/all_difference_cleaned_data.csv'
dataset.to_csv(output_file, index=False)

print(f"Data preprocessing complete. Cleaned dataset saved to '{output_file}'.")