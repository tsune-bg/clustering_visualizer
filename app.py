import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

from src import visualization, clustering


DATASETS_DIR = './data'


# タイトルの表示
st.title('Clustering Visualizer')

col1, col2 = st.columns(2)

with col1:
    csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    selected_file = st.selectbox('Select a dataset', csv_files)

with col2:
    file_path = os.path.join(DATASETS_DIR, selected_file)
    df = pd.read_csv(file_path)

    fig = visualization.plot_dataset(df)
    st.pyplot(fig, use_container_width=False)

col1, col2 = st.columns(2)

with col1:
    k = st.number_input('Number of cluster', min_value=2, max_value=5, value=2)
with col2:
    random_state = st.number_input('Seed of random number', min_value=0, max_value=100, value=42)

start_button = st.button("Start Animation")
if start_button:
    X = df[['x', 'y']].to_numpy()
    center_history, label_history = clustering.kmeans(X, k, random_state)
    anim = visualization.animation_kmeans(X, center_history, label_history)

    components.html(anim.to_jshtml(), height=1000, scrolling=True)
