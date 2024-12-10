import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# Define column names
# Define the column names
columns = [
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
    "blueWin",
]

# Load dataset
file_path = "match_data_v5.csv"  # Replace with the actual file path
df = pd.read_csv(file_path, header=None)

# Drop the last column (the extra 0)
df = df.iloc[:, :-1]

# Assign column names
df.columns = columns

# Drop the 'matchId' column as it's not numerical
df = df.drop(columns=["matchId"])

# Create a folder for the distribution plots
# Calculate the number of rows and columns for subplots
n_cols = 4  # Number of columns in the grid
n_rows = math.ceil(len(df.columns) / n_cols)

# Create a figure and axes for subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))

# Flatten axes for easier indexing
axes = axes.flatten()

# Plot each column's distribution
for i, column in enumerate(df.columns):
    ax = axes[i]
    ax.hist(df[column], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(column)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

# Hide any unused subplots
for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()
