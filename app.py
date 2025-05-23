
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess dataset
df = pd.read_csv('netflix_titles.csv', encoding='latin1')
df.drop(df.columns[12:26], axis=1, inplace=True)
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Unknown', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)
df['duration_min'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['duration_min'].fillna(df['duration_min'].median(), inplace=True)
df['duration'] = df['duration_min'].astype(int).astype(str) + " min"
df['date_added'] = df['date_added'].str.strip()
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df.dropna(subset=['title', 'description', 'listed_in'], inplace=True)

# Combine features for TF-IDF
df['combined_features'] = df['title'] + ' ' + df['description'] + ' ' + df['listed_in']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title):
    title = title.lower()
    if title not in df['title'].str.lower().values:
        return []
    idx = df[df['title'].str.lower() == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return df['title'].iloc[sim_indices].tolist()

# Streamlit UI
st.title("Netflix Show Recommender")
st.write("Enter a show or movie title, and we'll recommend similar content!")

user_input = st.text_input("Enter Title")
if user_input:
    recommendations = get_recommendations(user_input)
    if recommendations:
        st.subheader("You might also like:")
        for title in recommendations:
            st.write(f"- {title}")
    else:
        st.warning("Title not found. Please try another one.")
