# train_and_save_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv(r'C:\Users\smgop\Desktop\Task\Disease_prediction\improved_disease_dataset.csv')

# Encode target label
le = LabelEncoder()
df['N_disease'] = le.fit_transform(df['disease'])

# Feature matrix and target
X = df.iloc[:, :-2]  # exclude 'disease' and 'N_disease'
y = df['N_disease']
symptom_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model, label encoder, and column names
with open('model.pkl', 'wb') as f:
    pickle.dump((model, le, symptom_columns), f)

print("Model training complete. Saved to model.pkl")
