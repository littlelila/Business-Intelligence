import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

# Load your dataset (assuming it has headers)

df = pd.read_csv('../gold_cleaned_data_difference.csv')


# Ensure your dataset has no missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['blueWin'])
y = df['blueWin']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoostClassifier
model = XGBClassifier(
     n_estimators=500,       # Number of boosting rounds (trees)
    learning_rate=0.01,     # Step size shrinkage
    max_depth=6,            # Maximum tree depth for base learners
    subsample=1,          # Fraction of samples used for each tree
    colsample_bytree=0.6,   # Fraction of features used for each tree
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False # Suppresses a warning for older versions of XGBoost
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
log_loss_value = log_loss(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")

# Predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of blueWin = 1

# Brier Score
brier_score = brier_score_loss(y_test, y_pred_prob)
print(f"Brier Score: {brier_score}")

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
joblib.dump(model, 'lol_win_predictor_xgb.pkl')
