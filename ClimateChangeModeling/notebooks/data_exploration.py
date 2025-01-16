import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download NLTK resources (run once; can be commented out later)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load the dataset
dataset_path = 'D:/ClimateChangeModeling/data/climate_data.csv'  # Adjust the path as needed
data = pd.read_csv(dataset_path)

# Display basic information
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n", data.info())
print("\nDataset Description:\n", data.describe())

# Handle missing values
data['likesCount'] = data['likesCount'].fillna(data['likesCount'].mean())
data['commentsCount'] = data['commentsCount'].fillna(data['commentsCount'].mean())

# Encode categorical data
label_encoder = LabelEncoder()
data['ProfileName_Encoded'] = label_encoder.fit_transform(data['profileName'])

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Text preprocessing function
def preprocess_text(text):
    if pd.isnull(text):  # Handle NaN values
        return ""
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text into words
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Return the cleaned text
    return ' '.join(tokens)

# Apply preprocessing to the 'text' column
if 'text' in data.columns:  # Ensure the 'text' column exists
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    print("\nCleaned Text Column:\n", data[['text', 'cleaned_text']].head())
else:
    print("\nColumn 'text' not found in the dataset.")

# Sentiment analysis using VADER
if 'cleaned_text' in data.columns:
    data['sentiment'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    print("\nSentiment Scores:\n", data[['cleaned_text', 'sentiment']].head())
else:
    print("\nColumn 'cleaned_text' not found in the dataset.")

# Generate TF-IDF features from cleaned_text
tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Use top 100 features
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_text']).toarray()

# Combine numerical and text features
numeric_features = data[['commentsCount', 'sentiment']].values  # Numerical features
combined_features = np.hstack([numeric_features, tfidf_features])  # Combine numerical and TF-IDF features

# Define the target variable (labels)
labels = data['likesCount'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Check the shapes of the splits
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Initialize the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the values for the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")

# Visualize the predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Likes Count")
plt.ylabel("Predicted Likes Count")
plt.title("Actual vs Predicted Likes Count")
plt.show()

# Optional: Visualize the sentiment score distribution
plt.hist(data['sentiment'], bins=20, color='skyblue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Optional: Plot Feature Importances (For RandomForest)
importances = model.feature_importances_
features = ['commentsCount', 'sentiment'] + list(tfidf_vectorizer.get_feature_names_out())

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()

# Optional: Plot residuals (Actual vs Error)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')  # Line at y=0
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Likes Count')
plt.ylabel('Residuals')
plt.show()

# Save the updated dataset with cleaned text and sentiment scores
data.to_csv('cleaned_climate_data_with_sentiment.csv', index=False)
print("\nSaved the dataset with sentiment scores as 'cleaned_climate_data_with_sentiment.csv'")
