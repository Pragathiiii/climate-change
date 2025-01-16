# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler  # For scaling numeric features

# Load preprocessed data (this is the cleaned data after preprocessing step)
dataset_path = 'D:/ClimateChangeModeling/data/climate_data_cleaned.csv'  # Ensure this path matches your preprocessed data
data = pd.read_csv(dataset_path)

# Standardize or normalize numeric columns for model training
numeric_columns = ['LikesCount', 'CommentsCount']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Define features (X) and target variable (y)
X = data.drop(columns=['likesCount'])  # Features
y = data['likesCount']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
