Bank Customer Churn Prediction
This project aims to predict customer churn in a bank using a dataset containing various customer attributes. 
The model uses a Random Forest Classifier to predict whether a customer will exit (churn) or not.
The project includes data preprocessing, model training, evaluation, and visualization.

Dataset
the dataset is taken from kaggle :-
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction


The dataset used in this project is Churn_Modelling.csv, which contains the following columns:
RowNumber: Row index.
CustomerId: Unique identifier for each customer.
Surname: Customer's surname.
CreditScore: Customer's credit score.
Geography: Customer's country.
Gender: Customer's gender.
Age: Customer's age.
Tenure: Number of years the customer has been with the bank.
Balance: Account balance.
NumOfProducts: Number of products the customer has.
HasCrCard: Whether the customer has a credit card (1) or not (0).
IsActiveMember: Whether the customer is an active member (1) or not (0).
EstimatedSalary: Estimated salary of the customer.
Exited: Whether the customer churned (1) or not (0).
Project Structure
data_loading: Load the dataset and handle missing values.
data_preprocessing: Drop unnecessary columns and convert categorical variables into dummy/indicator variables.
data_visualization: Visualize the distribution of the target variable.
model_training: Split the data into training, validation, and test sets, and train a Random Forest Classifier.
model_evaluation: Evaluate the model using accuracy, confusion matrix, classification report, and ROC curve.

Installation
To run this project, ensure you have the following libraries installed:

numpy
pandas
matplotlib
seaborn
scikit-learn
You can install these libraries using pip:
pip install <library name>


Results
The model's performance is evaluated based on accuracy, confusion matrix, classification report, and ROC curve. 
These metrics provide a comprehensive understanding of how well the model predicts customer churn.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
