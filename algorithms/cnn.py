import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load your dataset (assuming it has headers)
df = pd.read_csv('cleaned_data.csv')

# Ensure your dataset has no missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['blueWin'])
y = df['blueWin']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # Input layer with 128 neurons
    Dropout(0.3),                                              # Dropout to prevent overfitting
    Dense(64, activation='relu'),                              # Hidden layer with 64 neurons
    Dropout(0.3),                                              # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')                             # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=50,             # Number of training iterations
                    batch_size=32,         # Number of samples per gradient update
                    validation_split=0.2,  # Use 20% of training data for validation
                    verbose=1)             # Print progress

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print performance metrics
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Log Loss: {log_loss(y_test, y_pred_proba):.4f}")

# Optional: Plot training history
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Save the trained model
model.save('lol_win_predictor_nn.h5')
