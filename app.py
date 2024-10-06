import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

from src import visualization, clustering


DATASETS_DIR = './data'


# タイトルの表示
st.title('Clustering Visualizer')

# セレクトボックスでファイルを選択
csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
selected_file = st.selectbox('Select a dataset', csv_files)

# CSV を読み込む
file_path = os.path.join(DATASETS_DIR, selected_file)
df = pd.read_csv(file_path)

# データセットの描画
fig = visualization.plot_dataset(df)
st.pyplot(fig, use_container_width=False)


k = st.number_input('Number of cluster', min_value=2, max_value=5)
random_state = st.number_input('Seed of random number', min_value=0, max_value=100)

start_button = st.button("Start Animation")
if start_button:
    X = df[['x', 'y']].to_numpy()
    center_history, label_history = clustering.kmeans(X, k, random_state)
    anim = visualization.animation_kmeans(X, center_history, label_history)

    components.html(anim.to_jshtml(), height=1000)
