import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("GermanCredit.csv")

# Encode categorical columns
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create model
model = DecisionTreeClassifier(criterion='entropy')

# 10-fold cross validation
scores = cross_val_score(model, X, y, cv=10)

# Print results
print("Cross validation accuracy for each fold:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean() * 100)