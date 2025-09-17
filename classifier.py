import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Ensure you have the necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
# Adjust the file path if your data is in a different location
train_df = pd.read_csv('train_data.txt', sep=':::', engine='python', names=['Title', 'Genre', 'Description'])

## Step 1: Data Preprocessing
# This part cleans the text data by removing special characters, stop words, and applying lemmatization.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A).lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

train_df['clean_description'] = train_df['Description'].apply(preprocess_text)

## Step 2: Feature Extraction
# Converts the cleaned text into numerical features using TF-IDF.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(train_df['clean_description'])
y = train_df['Genre']

## Step 3: Train-Test Split
# Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Step 4: Handle Imbalanced Data and Train the Model
# Use SMOTE to oversample the minority classes in the training data.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print the class distribution before and after resampling to see the effect of SMOTE.
print("Original dataset shape:", sorted(Counter(y_train).items()))
print("Resampled dataset shape:", sorted(Counter(y_resampled).items()))

# Train the Multinomial Naive Bayes model on the resampled data.
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

## Step 5: Evaluate the Model
# Make predictions on the original, non-resampled test set and evaluate the performance.
y_pred = model.predict(X_test)

print("\nAccuracy with SMOTE:", accuracy_score(y_test, y_pred))
print("\nClassification Report with SMOTE:\n", classification_report(y_test, y_pred))


def predict_genre(text):
    # Preprocess the input text using the same function as before
    processed_text = preprocess_text(text)
    # Convert the preprocessed text into TF-IDF features
    text_features = tfidf_vectorizer.transform([processed_text])
    # Make a prediction using the trained model
    prediction = model.predict(text_features)
    return prediction[0]

# --- Example Usage ---
# Use a movie plot summary that your model has never seen before.
new_movie_plot = "A group of astronauts on a deep space mission discover a hostile alien life form on board their ship, leading to a terrifying fight for survival."
predicted_genre = predict_genre(new_movie_plot)

print(f"\nThe plot: {new_movie_plot}")
print(f"Predicted genre: {predicted_genre}")