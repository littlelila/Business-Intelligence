import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the new column names (updated with derived attributes)
columns = [
    "blueTeamControlWardsPlaced",
    "blueTeamWardsPlaced",
    "blueTeamTurretPlatesDestroyed",
    "blueTeamFirstBlood",
    "blueTeamMinionsKilled",
    "blueTeamJungleMinions",
    "redTeamControlWardsPlaced",
    "redTeamWardsPlaced",
    "redTeamTurretPlatesDestroyed",
    "redTeamMinionsKilled",
    "redTeamJungleMinions",
    "blueWin",
    "GoldDifference",
    "XpDifference",
    "DamageDifference",
    "KillDifference",
    "DragonKillDifference",
    "HeraldKillDifference",
    "TowerDifference",
    "InhibitorDifference"
]

# Load the dataset
file_path = "cleaned_data_difference.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Assign column names
df.columns = columns

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Compute the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Automatically adjust the layout to fit everything
plt.title("Correlation Matrix Heatmap")
plt.show()
