import streamlit as st
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import requests
from difflib import SequenceMatcher
#from retrying import retry
#import logging

#logging.basicConfig(level=logging.DEBUG)

with open('book_pivot.pkl', 'rb') as pivot_file:
    book_pivot = pickle.load(pivot_file)

with open('book_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

unique_books_names = pd.Series(book_pivot.index)

GOOGLE_BOOKS_API_KEY = st.secrets["api_key"]

#@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
#def make_request(url):
    #return requests.get(url)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_google_books_info(book_title, target_language='en'):

    book_title = book_title.replace(" ", "%20")

    #url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}&key={GOOGLE_BOOKS_API_KEY}&country=US"

    #response = make_request(url)

    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}&key={GOOGLE_BOOKS_API_KEY}"

    headers = {'x-forwarded-for': '89.44.80.91'}

    response = requests.get(url, headers=headers)
    
    #if response.status_code != 200:
        #logging.debug(f"Response: {response.status_code}, {response.text}")
        #logging.debug(f"Request URL: {url}")
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('items', [])

        top3_preview_links = [item['volumeInfo'].get('previewLink', '') for item in items[:3]]

        if items:
            filtered_items = [item for item in items if item['volumeInfo'].get('language', '') == target_language]

            sorted_items = sorted(filtered_items, key=lambda item: similar(book_title, item['volumeInfo']['title']), reverse=True)

            if sorted_items:
                item = sorted_items[0]['volumeInfo']

                genres = item.get('categories', ['Genre not available'])[:5]
                description = item.get('description', 'Description not available')[:200] + "..."

                result = {
                    'title': (item.get('title', 'Title not available')),
                    'authors': ', '.join(item.get('authors', ['Author not available'])),
                    'genres': ', '.join(genres),
                    'description': description,
                    'preview_link': item.get('previewLink', ''),
                    'top3_preview_links': top3_preview_links,
                    'small_thumbnail': item.get('imageLinks', {}).get('smallThumbnail', '')
                }

                return result

    return None

def recommend_books(book_title, pivot_data, model, num_recommendations=10):
    try:
        row_position = pivot_data.index.get_loc(book_title)
    except KeyError:
        st.error(f"Book title '{book_title}' not found in the DataFrame.")
        return

    distances, suggestions = model.kneighbors(pivot_data.iloc[row_position, :].values.reshape(1, -1), n_neighbors=num_recommendations)

    st.markdown("""
        <style>
            button[title="View fullscreen"] {
                visibility: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"## Top {num_recommendations} Recommendations:")
    for i in range(num_recommendations):
        recommended_book_title = pivot_data.index[suggestions[0, i]]
        distance_to_recommended_book = distances[0][i]

        google_books_info = get_google_books_info(recommended_book_title)

        st.markdown(f"<h4>{i + 1}. {recommended_book_title}<br> (Distance to a book: {distance_to_recommended_book:.2f})</h4>", unsafe_allow_html=True)
        if google_books_info:
            small_thumbnail = google_books_info.get('small_thumbnail', 'Thumbnail not available')
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.write(f"   Title: {google_books_info.get('title', 'Title not available')}")
                st.write(f"   Authors: {google_books_info.get('authors', 'Authors not available')}")
                st.write(f"   Genres: {google_books_info.get('genres', 'Genres not available')}")
                st.write(f"   Description: {google_books_info.get('description', 'Description not available')}")
                
                preview_link = google_books_info.get('preview_link', 'Preview Link not available')
                st.markdown(f"Preview Link: <a href='{preview_link}' target='_blank'>Google Books Preview</a>", unsafe_allow_html=True)

                with st.expander(f"Another 3 possible books links", expanded=False):
                    for j, another_preview_link in enumerate(google_books_info.get('top3_preview_links', [])):
                        st.markdown(f"Book Link Number {j + 1}: <a href='{another_preview_link}' target='_blank'>Google Books Preview</a>", unsafe_allow_html=True)

            if small_thumbnail and small_thumbnail != 'Thumbnail not available':
                col2.image(small_thumbnail, caption="Book image", use_column_width=False, width=75)
            else:
                st.write("   Thumbnail not available")

        else:
            st.write("   Book information not available")

st.title("Book Recommendation App")

book_title_to_find = st.selectbox("Enter the book title:", unique_books_names, help="Type the book title")

if st.button("Get Recommendations"):
    st.markdown(f"### Recommendations for '{book_title_to_find}':")
    recommend_books(book_title_to_find, book_pivot, model)
