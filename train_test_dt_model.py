import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sound_spectrum_analysis import extract_features  # Replace with your actual feature extraction function

# Define the training directory and labels
base_dir = "trainingAudio"
labels = ["dogBark", "microwaves", "smokeAlarms"]

# Prepare data
X = []  # Feature vectors
y = []  # Labels

# Extract features from audio files
for label in labels:
    label_dir = os.path.join(base_dir, label)
    if not os.path.exists(label_dir):
        print(f"Directory not found: {label_dir}")
        continue

    for file_name in os.listdir(label_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(label_dir, file_name)
            features = extract_features(file_path)


            if features is not None:

                # Combine these features with the existing ones
                combined_features = np.hstack((features))
                X.append(combined_features)
                y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)



# Check if we have enough data
if len(X) == 0:
    print("No valid training data found. Please check the audio files and feature extraction.")
    exit()

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and validation
train_ratio = 0.7
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - train_ratio), stratify=y, random_state=42)

# Decision Tree: Iterate over hyperparameters
best_dt_model = None
best_dt_val_accuracy = 0
print("\nTesting Decision Tree Classifier with various hyperparameters...")
for max_depth in [5, 10, 15, None]:
    for min_samples_split in [2, 5, 10]:
        print(f"Testing Decision Tree: max_depth={max_depth}, min_samples_split={min_samples_split}")
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate accuracy
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        print(f"Decision Tree - Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Track the best model
        if val_accuracy > best_dt_val_accuracy:
            best_dt_val_accuracy = val_accuracy
            best_dt_model = model

if best_dt_model:
    print(f"\nBest Decision Tree: max_depth={best_dt_model.get_params()['max_depth']}, min_samples_split={best_dt_model.get_params()['min_samples_split']}")
    print(f"Best Decision Tree Validation Accuracy: {best_dt_val_accuracy * 100:.2f}%")
    joblib.dump(best_dt_model, "decision_tree_classifier.pkl")
    print("Best Decision Tree model saved as decision_tree_classifier.pkl")

# Random Forest: Train and evaluate
print("\nTraining Random Forest Classifier...")
best_rf_model = None
best_rf_val_accuracy = 0

# Iterate over hyperparameters for Random Forest
for n_estimators in [50, 100, 150]:
    for max_depth in [5, 10, 15, None]:
        print(f"Testing Random Forest: n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate accuracy
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        print(f"Random Forest - Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Track the best model
        if val_accuracy > best_rf_val_accuracy:
            best_rf_val_accuracy = val_accuracy
            best_rf_model = model

if best_rf_model:
    print(f"\nBest Random Forest: n_estimators={best_rf_model.get_params()['n_estimators']}, max_depth={best_rf_model.get_params()['max_depth']}")
    print(f"Best Random Forest Validation Accuracy: {best_rf_val_accuracy * 100:.2f}%")
    joblib.dump(best_rf_model, "random_forest_classifier.pkl")
    print("Best Random Forest model saved as random_forest_classifier.pkl")
