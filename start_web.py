import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load data and model
with open('book_pivot.pkl', 'rb') as pivot_file:
    book_pivot = pickle.load(pivot_file)

with open('book_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load unique book names for autocomplete
unique_books_names = pd.Series(book_pivot.index)

# Function to recommend books
def recommend_books(book_title, pivot_data, model, num_recommendations=10):
    try:
        row_position = pivot_data.index.get_loc(book_title)
    except KeyError:
        st.error(f"Book title '{book_title}' not found in the DataFrame.")
        return

    distances, suggestions = model.kneighbors(pivot_data.iloc[row_position, :].values.reshape(1, -1), n_neighbors=num_recommendations)

    st.markdown(f"## Top {num_recommendations} Recommendations:")
    for i in range(num_recommendations):
        recommended_book_title = pivot_data.index[suggestions[0, i]]
        distance_to_recommended_book = distances[0][i]
        st.write(f"{i + 1}. {recommended_book_title} (Distance: {distance_to_recommended_book:.2f})")

# Streamlit App
st.title("Book Recommendation App")

# Input for book title with autocomplete in the center
book_title_to_find = st.selectbox("Enter the book title:", unique_books_names, help="Type the book title")

# Button to trigger recommendations
if st.button("Get Recommendations"):
    st.markdown(f"### Recommendations for '{book_title_to_find}':")
    recommend_books(book_title_to_find, book_pivot, model)

# Add additional features, explanations, or information as needed
