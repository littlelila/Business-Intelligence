import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

class SklearnCompatibleXGBClassifier(XGBClassifier, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __sklearn_tags__(self):
        return {'multioutput': False}


# Load your dataset (assuming it has headers)
df = pd.read_csv('../cleaned_data.csv')

# Ensure your dataset has no missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['blueWin'])
y = df['blueWin']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 300, 500],       # Number of boosting rounds (trees)
    'learning_rate': [0.001, 0.01, 0.1, 1],    # Step size shrinkage
    'max_depth': [3, 6, 9],                # Maximum tree depth for base learners
}

# Initialize the model
# Replace XGBClassifier with SklearnCompatibleXGBClassifier
xgb = SklearnCompatibleXGBClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',  # You can also use 'neg_log_loss' or other metrics
    cv=5,                # 5-fold cross-validation
    verbose=1,
    n_jobs=-1            # Use all available cores
)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best CV Score:", best_score)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
log_loss_value = log_loss(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Log Loss: {log_loss_value:.4f}")

# Feature importance using permutation importance
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)

# Extract and sort feature importances 
sorted_idx = perm_importance.importances_mean.argsort()

# Plot feature importances
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.show()

# Save the best model
joblib.dump(best_model, 'lol_win_predictor_xgb_best.pkl')
