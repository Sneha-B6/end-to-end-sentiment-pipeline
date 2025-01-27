# data_collection.py
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Try to load the IMDB dataset using Hugging Face's datasets library
try:
    dataset = load_dataset('imdb')

    # Verify the dataset size and ensure it loaded correctly
    print("Dataset loaded successfully!")

    # Print the structure of the dataset (to see what it contains)
    print(f"Dataset Structure: {dataset}")
    
    # Print the sizes of train and test sets
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    # Optionally, you can inspect a few rows to make sure it looks correct
    print("\nSample data from the train set:")
    print(dataset['train'][0])

except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
