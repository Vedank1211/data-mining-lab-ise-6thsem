import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score

# Load data
data = pd.read_csv("GermanCredit.csv")

# Encode categorical features
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# FIX 1: Match the column names exactly to your provided list
selected_columns = [
    'credit_history', 
    'purpose', 
    'employment_duration', # was 'employement'
    'other_debtors',       # was 'debtors'
    'housing', 
    'present_residence'    # was 'residence'
]

X = data[selected_columns]
y = data['credit_risk']    # FIX 2: Changed 'class' to 'credit_risk'

# FIX 3: Changed 'empty' to 'gini' (valid criterion)
model = DecisionTreeClassifier(criterion='gini')

# FIX 4: Changed 'cross_val_scores' to 'cross_val_score'
scores = cross_val_score(model, X, y, cv=10)

print("Accuracy for each fold:")
print(scores)
print("\nAverage accuracy:")
# FIX 5: Added () to call the mean function
print(f"{scores.mean() * 100:.2f}%")