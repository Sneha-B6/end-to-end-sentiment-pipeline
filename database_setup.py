# database_setup.py
import sqlite3
from datasets import load_dataset

# Try to load the IMDB dataset using Hugging Face's datasets library
try:
    dataset = load_dataset('imdb')

    # Verify the dataset size and ensure it loaded correctly
    print("Dataset loaded successfully!")

    # Connect to SQLite database (create if doesn't exist)
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()

    # Drop the table if it exists (to reset the structure)
    cursor.execute('DROP TABLE IF EXISTS imdb_reviews')

    # Create the table with the correct structure
    cursor.execute('''
    CREATE TABLE imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT,
        sentiment TEXT
    )
    ''')

    # Confirm the table creation
    print("Table 'imdb_reviews' is ready!")

    # Insert dataset into the SQLite database
    for idx, review in enumerate(dataset['train']):
        review_text = review['text']
        sentiment = 'Positive' if review['label'] == 1 else 'Negative'

        cursor.execute('''
        INSERT INTO imdb_reviews (review_text, sentiment) VALUES (?, ?)
        ''', (review_text, sentiment))

        if idx % 1000 == 0:  # Commit every 1000 records to avoid memory issues
            conn.commit()

    # Commit any remaining records and close the connection
    conn.commit()
    print("Dataset inserted into the database successfully!")

    # Optionally, fetch and display the first 5 rows from the database
    cursor.execute('SELECT * FROM imdb_reviews LIMIT 5')
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    conn.close()

except Exception as e:
    print(f"An error occurred: {e}")
