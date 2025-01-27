import sqlite3
import pandas as pd
import string
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to SQLite database
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Load the dataset from SQLite database into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM imdb_reviews", conn)

# A. Data Cleaning
# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Remove null values
df = df.dropna(subset=['review_text'])

# Preprocess the text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    return text

# Apply cleaning function to the review_text column
df['cleaned_review_text'] = df['review_text'].apply(clean_text)

# Show a preview of the cleaned data
print(f"Cleaned Data Preview:\n{df.head()}")

# B. Exploratory Data Analysis (EDA)
# 1. Distribution of sentiments
sentiment_counts = df['sentiment'].value_counts()

# Plot sentiment distribution (Bar chart)
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 2. Average review length for each sentiment
df['review_length'] = df['cleaned_review_text'].apply(len)

# Calculate average review length by sentiment
avg_review_length = df.groupby('sentiment')['review_length'].mean()

# Plot average review length for each sentiment (Bar chart)
avg_review_length.plot(kind='bar', title='Average Review Length by Sentiment', color=['red', 'green'])
plt.ylabel('Average Review Length')
plt.show()

# C. Insert the cleaned data back into the SQLite table
# Create a function to insert the cleaned data into the database

def insert_cleaned_data(df):
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE imdb_reviews
            SET review_text = ?, sentiment = ?
            WHERE id = ?
        ''', (row['cleaned_review_text'], row['sentiment'], row['id']))
    conn.commit()

# Insert cleaned data back into the table
insert_cleaned_data(df)

# Confirm the data has been updated in the table
cursor.execute('SELECT * FROM imdb_reviews LIMIT 5')
print(cursor.fetchall())

# Close the database connection
conn.close()
