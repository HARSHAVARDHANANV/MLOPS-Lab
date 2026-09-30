import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Create a simple dataset
data = {
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Study_Hours': [4.5, 7.0, np.nan, 3.2, 8.5, np.nan],
    'Attendance': [92, 85, 78, np.nan, 95, 90],
    'Passed': [1, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. Fix missing values with the column average
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].mean())
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())

# 3. Convert text to numbers
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# 4. Split into training and testing sets (80% train, 20% test)
X = df[['Gender', 'Study_Hours', 'Attendance']]
y = df['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data cleaned and split successfully!")