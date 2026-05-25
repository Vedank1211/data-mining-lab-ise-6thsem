import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score

# 1. Load the data
data = pd.read_csv("GermanCredit.csv")

# 2. Fix the LabelEncoder loop
label_encoder = LabelEncoder()
for column in data.columns:  
    if data[column].dtype == 'object':  
        data[column] = label_encoder.fit_transform(data[column])

# 3. Split features and target
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]

# 4. Initialize the model
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=10)

# 5. Evaluate using Cross-Validation
scores = cross_val_score(model, X, y, cv=10) 

# 6. Fit the model on the full dataset
model.fit(X, y)  

# 7. Print results
print("Cross-validation scores for each fold:")
print(scores)
print("\nAverage accuracy:")
print(f"{scores.mean() * 100:.2f}%")
