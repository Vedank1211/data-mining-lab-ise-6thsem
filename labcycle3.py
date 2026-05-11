# Install required packages
# pip install pandas scikit-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Load dataset
data = pd.read_csv("GermanCredit.csv")

print("Dataset Sample")
print(data.head())

# Encode categorical columns
label_encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train Decision Tree model
model = DecisionTreeClassifier(criterion='entropy')

model.fit(X, y)

# Export tree rules
tree_rules = export_text(model, feature_names=list(X.columns))

print("\nDecision Tree Model:\n")
print(tree_rules)