import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib

# Load your dataset (assuming it has headers)
df = pd.read_csv('../all_difference_cleaned_data.csv')

# Ensure your dataset has no missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['blueWin'])
y = df['blueWin']  # Already binary, no need for scaling

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBRegressor
model = XGBRegressor(
    n_estimators=500,       # Number of boosting rounds (trees)
    learning_rate=0.01,    # Step size shrinkage
    max_depth=3,            # Maximum tree depth for base learners
    subsample=0.6,            # Fraction of samples used for each tree
    colsample_bytree=0.6,   # Fraction of features used for each tree
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Metrics for regression
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Optional: If you want to interpret probabilities as classifications (e.g., >0.5 means win):
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = (y_test == y_pred_class).mean()
print(f"Classification Accuracy (Threshold 0.5): {accuracy:.4f}")

# Feature importance using permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Extract and sort feature importances
sorted_idx = perm_importance.importances_mean.argsort()

# Plot feature importances
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.show()

# Save the trained model
joblib.dump(model, 'lol_win_predictor_xgb.pkl')
