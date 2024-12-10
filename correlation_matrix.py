import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the column names
columns = [
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

# Load the dataset
file_path = "cleaned_data.csv"  # Replace with your file path
df = pd.read_csv(file_path, header=None)


# Drop the last column (the extra 0)
#df = df.iloc[:, :-1]


# Assign column names
df.columns = columns

# Drop the 'matchId' column as it's not numerical
#df = df.drop(columns=["matchId"])

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Compute the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# Adjust layout to prevent labels from being cut off
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Automatically adjust the layout to fit everything
plt.title("Correlation Matrix Heatmap")
plt.show()