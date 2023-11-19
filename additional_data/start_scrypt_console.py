import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

with open('book_pivot.pkl', 'rb') as pivot_file:
    book_pivot = pickle.load(pivot_file)

with open('book_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def recommend_books(book_title, pivot_data, model, num_recommendations=10):
    try:
        row_position = pivot_data.index.get_loc(book_title)
    except KeyError:
        print(f"Book title '{book_title}' not found in the DataFrame.")
        return

    distances, suggestions = model.kneighbors(pivot_data.iloc[row_position, :].values.reshape(1, -1), n_neighbors=num_recommendations)

    print(f"\nTop {num_recommendations} Recommendations:")
    for i in range(num_recommendations):
        recommended_book_title = pivot_data.index[suggestions[0, i]]
        distance_to_recommended_book = distances[0][i]
        print(f"{i + 1}. {recommended_book_title} (Distance: {distance_to_recommended_book:.2f})")    

book_title_to_find = "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)"
recommend_books(book_title_to_find, book_pivot, model)