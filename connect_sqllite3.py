# data_collection.py
import sqlite3
from datasets import load_dataset

# Try to load the IMDB dataset using Hugging Face's datasets library
try:
    dataset = load_dataset('imdb')

    # Verify the dataset size and ensure it loaded correctly
    print("Dataset loaded successfully!")

    # Print the sizes of train and test sets
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    # Connect to SQLite database (create if doesn't exist)
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()

    # Create the table to store dataset if not already created
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        label INTEGER
    )
    ''')

    # Confirm the table creation
    print("Table 'imdb_reviews' is ready!")

    # Insert dataset into the SQLite database
    for idx, review in enumerate(dataset['train']):
        text = review['text']
        label = review['label']
        cursor.execute('''
        INSERT INTO imdb_reviews (text, label) VALUES (?, ?)
        ''', (text, label))

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
