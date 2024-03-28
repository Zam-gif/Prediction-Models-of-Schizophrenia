import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the training and test datasets
train_df = pd.read_csv('path_to_training_data.csv')  # path to training dataset
test_df = pd.read_csv('path_to_test_data.csv')  # Path to test dataset

# Extract features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train).toarray()
X_test_encoded = encoder.transform(X_test).toarray()

# Initialize models
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(penalty='l2', C=1e8, solver='liblinear')
}

# Parameter grid for Random Forest
n_estimators_options = [10, 100, 1000, 10000]
max_depth_options = [2, 5, 10, 20, 50, 100]
max_features = [7, 7, 18, 37]


# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    train_pred_proba = model.predict_proba(X_train_encoded)[:, 1]
    auc_score = roc_auc_score(y_train, train_pred_proba)
    accuracy = accuracy_score(y_train, model.predict(X_train_encoded))
    results[name] = {'model': model, 'auc': auc_score, 'accuracy': accuracy}

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=n_SNPs, random_state=42)
        rf_model.fit(X_train_encoded, y_train)
        train_pred_proba = rf_model.predict_proba(X_train_encoded)[:, 1]
        auc_score = roc_auc_score(y_train, train_pred_proba)
        accuracy = accuracy_score(y_train, rf_model.predict(X_train_encoded))
        model_name = f'Random Forest ({n_estimators}, {max_depth})'
        results[model_name] = {'model': rf_model, 'auc': auc_score, 'accuracy': accuracy}

# Identify best model based on AUC
#best_model_name = max(results.keys(), key=lambda name: results[name]['auc'])
#best_model_info = results[best_model_name]
#best_model = best_model_info['model']

# Identify best model based on Accuracy
best_model_name = max(results.keys(), key=lambda name: results[name]['accuracy'])
best_model_info = results[best_model_name]
best_model = best_model_info['model']


# Evaluate best model on test data
y_pred = best_model.predict(X_test_encoded)
y_pred_proba = best_model.predict_proba(X_test_encoded)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
test_accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Plotting functions
def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Display training performance
print("Training Data - Model Performance:")
for name, info in results.items():
    print(f"{name} - ROC AUC: {info['auc']:.1%}, Accuracy: {info['accuracy']:.1%}")

# Display best model's test performance
print("\nTest Data - Best Model Performance:")
print(f"Best Model: {best_model_name}")
print(f" Test AUC: {test_auc:.1%}, Accuracy: {test_accuracy:.1%}")
print(f"Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%}")

# Plot the confusion matrix for the best model
#plot_conf_matrix(y_test, y_pred_test)
