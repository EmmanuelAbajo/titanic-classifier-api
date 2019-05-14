# Import dependencies
import pandas as pd
import numpy as np
import joblib

# Load the dataset in a dataframe object and include only four features as mentioned
df = pd.read_csv('titanic_data.csv')
features = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df_ = df[features]

# Data Preprocessing
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
target = 'Survived'
x = df_ohe[df_ohe.columns.difference([target])]
y = df_ohe[target]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
