import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# File paths for training
train_file_paths = [
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X1_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X2_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X3_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X4_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X5_01.csv'
]

# File paths for validation
validate_file_paths = [
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X1_fold_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X2_fold_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X3_fold_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X4_fold_01.csv',
    'C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/X5_fold_01.csv'
]

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Hyperparameters for Logistic Regression and Random Forest
log_reg_params = [1e-8, 1e-4, 1]  # Logistic regression hyperparameters
n_estimators_options = [10, 100, 1000, 10000]  # Random forest hyperparameters
max_depth_options = [2, 5, 10, 20, 50, 100]  # Random forest max depths

# Initialize dictionaries to store results
results = {
    'Naive Bayes': {'accuracy': []},
    'Logistic Regression': {'accuracy': {C: [] for C in log_reg_params}},
    'Random Forest': {'accuracy': {(n_estimators, max_depth): [] for n_estimators in n_estimators_options for max_depth in max_depth_options}}
}

# Perform cross-validation
for i in range(5):
    # Load training and validation datasets
    X_train = pd.read_csv(train_file_paths[i]).iloc[:, :-1]
    y_train = pd.read_csv(train_file_paths[i]).iloc[:, -1]
    X_val = pd.read_csv(validate_file_paths[i]).iloc[:, :-1]
    y_val = pd.read_csv(validate_file_paths[i]).iloc[:, -1]

    # One-hot encode categorical variables
    X_train_encoded = encoder.fit_transform(X_train).toarray()
    X_val_encoded = encoder.transform(X_val).toarray()

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_encoded, y_train)
    y_val_pred_nb = nb_model.predict(X_val_encoded)
    results['Naive Bayes']['accuracy'].append(accuracy_score(y_val, y_val_pred_nb))

    # Logistic Regression with different regularization strengths
    for C in log_reg_params:
        log_reg_model = LogisticRegression(C=C, solver='liblinear', random_state=42)
        log_reg_model.fit(X_train_encoded, y_train)
        y_val_pred_lr = log_reg_model.predict(X_val_encoded)
        results['Logistic Regression']['accuracy'][C].append(accuracy_score(y_val, y_val_pred_lr))

    # Random Forest with different hyperparameters
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_model.fit(X_train_encoded, y_train)
            y_val_pred_rf = rf_model.predict(X_val_encoded)
            results['Random Forest']['accuracy'][(n_estimators, max_depth)].append(accuracy_score(y_val, y_val_pred_rf))

# Calculate average accuracy for each model and hyperparameter
avg_accuracy = {
    'Naive Bayes': np.mean(results['Naive Bayes']['accuracy']),
    'Logistic Regression': {C: np.mean(acc) for C, acc in results['Logistic Regression']['accuracy'].items()},
    'Random Forest': {(n_estimators, max_depth): np.mean(acc) for (n_estimators, max_depth), acc in results['Random Forest']['accuracy'].items()}
}

# Print overall accuracies for each model and hyperparameter
print("Overall Accuracies:")
print(f"Naive Bayes: {avg_accuracy['Naive Bayes']:.1%}")

for C, acc in avg_accuracy['Logistic Regression'].items():
    print(f"Logistic Regression with C={C}: {acc:.1%}")

for (n_estimators, max_depth), acc in avg_accuracy['Random Forest'].items():
    print(f"Random Forest with n_estimators={n_estimators} and max_depth={max_depth}: {acc:.1%}")

# Select the best model and hyperparameters based on the highest average accuracy
best_model_type = 'Naive Bayes'
best_accuracy = avg_accuracy['Naive Bayes']
best_hyperparams = None

for C, acc in avg_accuracy['Logistic Regression'].items():
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_type = 'Logistic Regression'
        best_hyperparams = {'C': C}

for (n_estimators, max_depth), acc in avg_accuracy['Random Forest'].items():
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_type = 'Random Forest'
        best_hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth}

print(f"\nBest Model: {best_model_type} with Accuracy: {best_accuracy:.1%}")
if best_hyperparams:
    print(f"Hyperparameters: {best_hyperparams}")

# Train the best model with the selected hyperparameters on the full training data
train_df = pd.read_csv('C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/EAM_37SNP_training.csv')
X_train_final = train_df.iloc[:, :-1]
y_train_final = train_df.iloc[:, -1]
X_train_encoded_final = encoder.fit_transform(X_train_final).toarray()

if best_model_type == 'Logistic Regression':
    best_model = LogisticRegression(C=best_hyperparams['C'], solver='liblinear', random_state=42)
elif best_model_type == 'Random Forest':
    best_model = RandomForestClassifier(n_estimators=best_hyperparams['n_estimators'], max_depth=best_hyperparams['max_depth'], random_state=42)
else:
    best_model = GaussianNB()

best_model.fit(X_train_encoded_final, y_train_final)

# Print Training Accuracy and AUC
y_train_pred = best_model.predict(X_train_encoded_final)
y_train_pred_proba = best_model.predict_proba(X_train_encoded_final)[:, 1]
train_auc = roc_auc_score(y_train_final, y_train_pred_proba)
train_accuracy = accuracy_score(y_train_final, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.1%}, AUC: {train_auc:.1%}")

# Load and evaluate the best model on the test data
test_df = pd.read_csv('C:/Users/zamart.ramazanova/Documents/EA_M_5CV/5CV_07.06.2024/data_5CV_csv/EAM_37SNP_test.csv')
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
X_test_encoded = encoder.transform(X_test).toarray()

y_pred_test = best_model.predict(X_test_encoded)
y_pred_proba_test = best_model.predict_proba(X_test_encoded)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Display Test Evaluation Results
print(f"Test Accuracy: {test_accuracy:.1%}, AUC: {test_auc:.1%}")
print(f"Sensitivity: {sensitivity:.1%}, Specificity: {specificity:.1%}")

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

# Plot the confusion matrix for the best model on the test data
plot_conf_matrix(y_test, y_pred_test)
