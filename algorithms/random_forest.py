import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (assuming it has headers)
df = pd.read_csv('../difference_combined_scaled_cleaned_data.csv')

#df = df.drop(columns=["matchId"])

# Ensure your dataset has no missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['blueWin'])
y = df['blueWin']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)


# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    bootstrap=True,
    random_state=69      # Ensures reproducibility
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
log_loss_value = log_loss(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")

# Feature importance using permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Extract and sort feature importances
sorted_idx = perm_importance.importances_mean.argsort()

# Plot feature importances
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.show()

# Optional: Save the trained model
import joblib
joblib.dump(model, 'lol_win_predictor_rf.pkl')