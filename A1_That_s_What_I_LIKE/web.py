import streamlit as st
import numpy as np
from gensim.models import KeyedVectors

@st.cache_resource
def load_glove_model():
    glove_file = 'glove.6B.100d.txt'  # Update this path to the location of your GloVe file
    model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    return model

def get_top_similar_words(query, model, top_n=10):
    try:
        # Compute the similarity using the dot product
        similar_words = model.most_similar(query, topn=top_n)
        return similar_words
    except KeyError:
        return f"'{query}' not found in the vocabulary."

def main():
    st.title("Search Query Similarity Using Word Embeddings")
    st.subheader("Ponkrit Kaewsawee St124960")
    query = st.text_input("Enter your search query:")

    model = load_glove_model()

    if query:
        result = get_top_similar_words(query.lower(), model, top_n=10)

        if isinstance(result, list):
            st.write(f"Top 10 similar words to '{query}':")
            for word, score in result:
                st.write(f"{word}: {score:.4f}")
        else:
            st.write(result)

if __name__ == "__main__":
    main()
