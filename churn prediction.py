import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
# Load dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('C:\\Users\\SURYA VARMA\\OneDrive\\Desktop\\Bank Customer Churn Prediction\\Churn_Modelling.csv').dropna(axis=1)

# Drop unnecessary columns
data.drop(columns=['Surname', 'RowNumber', 'CustomerId'], inplace=True)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Geography', 'Gender'])

# Display dataset information
data.info()

# Visualize the target variable distribution
churn_count = data['Exited'].value_counts()
temp_df = pd.DataFrame({
    'Exited': churn_count.index,
    'Counts': churn_count.values
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Exited', y='Counts', data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Split data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)

# Evaluate model on validation set
preds = rf_model.predict(X_val)
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
print(f"Accuracy on validation data by Random Forest Classifier: {accuracy_score(y_val, preds) * 100}")

# Confusion Matrix for validation set
cf_matrix = confusion_matrix(y_val, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Validation Data")
plt.show()

# Predict on test dataset
y_pred = rf_model.predict(X_test)
print(f"Accuracy of the model on test dataset: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix for test set
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Classification Report for test set
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')

# ROC Curve and AUC for test set
y_proba = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
# Evaluate model on validation set
preds = rf_model.predict(X_val)
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
print(f"Accuracy on validation data by Random Forest Classifier: {accuracy_score(y_val, preds) * 100}")
# Predict on test dataset
y_pred = rf_model.predict(X_test)
print(f"Accuracy of the model on test dataset: {accuracy_score(y_test, y_pred)}")
