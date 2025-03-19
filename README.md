# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and Load Data: Import required libraries and load the dataset.
2. Preprocess Data: Remove unwanted columns, handle missing and duplicate values, and encode categorical variables.
3. Train Model: Split the data, train a logistic regression model, and make predictions.
4. Evaluate and Display Results: Calculate accuracy, display the classification report, and show the confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Jayakumar B
RegisterNumber: 212223040073
*/
```
```PY
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("d:/chrome downloads/Placement_Data.csv")
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

# Check for Missing and Duplicate Values
data1.isnull()
data1.duplicated().sum()

# Encode Categorical Variables
le = LabelEncoder()
for col in ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]:
    data1[col] = le.fit_transform(data1[col])

# Define Features and Target
x = data1.iloc[:, :-1]
y = data1["status"]

# Split Dataset into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train Logistic Regression Model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report1)

# Predict Placement
prediction = lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
print("Prediction:", prediction)
```
## Output:

Accuracy and Classification report:

![image](https://github.com/user-attachments/assets/9ae57e19-d1e0-4db6-9ffe-c53d99bb5399)

Prediction value for given input:

![image](https://github.com/user-attachments/assets/b6d1997c-5b48-4f7b-a09a-b559eae7a8cd)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
