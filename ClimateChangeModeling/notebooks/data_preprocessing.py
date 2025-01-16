import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
dataset_path = 'D:/ClimateChangeModeling/data/climate_data.csv'  # Adjust path if needed
data = pd.read_csv(dataset_path)

# Handle missing values (example: filling with mean for numeric columns)
data['likesCount'] = data['likesCount'].fillna(data['likesCount'].mean())
data['commentsCount'] = data['commentsCount'].fillna(data['commentsCount'].mean())

# Encode categorical data (ProfileName)
label_encoder = LabelEncoder()
data['ProfileName_Encoded'] = label_encoder.fit_transform(data['profileName'])

# Standardize numeric features
numeric_columns = ['likesCount', 'commentsCount']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Preview scaled data
print("\nScaled Data Head:\n", data.head())
