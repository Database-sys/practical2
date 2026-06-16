implement stochastic gradiant boosting algorithm on a data set of your choice
write a prolog program for solving tic-tac-toe problem
execute random forest classisification algorithm on any dataset of your choice





# Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)

# --------------------------------------------------
# 2. Split into train and test sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Create Stochastic Gradient Boosting model
# --------------------------------------------------
sgb = GradientBoostingClassifier(
    n_estimators=200,      
    learning_rate=0.05,  
    max_depth=3,          
    subsample=0.8,        
    random_state=42
)

# --------------------------------------------------
# 4. Train model
# --------------------------------------------------
sgb.fit(X_train, y_train)

# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------
y_pred = sgb.predict(X_test)

# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))























from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X = data.data
y = data.target
print("Dataset shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


sgb = GradientBoostingClassifier(
n_estimators=200,      
    learning_rate=0.05,  
    max_depth=3,          
    subsample=0.8,        
    random_state=42
)






sgb.fit(X_train, y_train)

y_pred = sgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))