import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from src import visualization


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

# スタートボタンを作成
start_button = st.button("Start Animation")

if start_button:
    num_frames = 10
    plot_placeholder = st.empty()

    for i in range(num_frames):
        fig = visualization.plot_dataset(df)
        fig.suptitle(f"Frame: {i}")
        plot_placeholder.pyplot(fig, use_container_width=False)

        time.sleep(0.75)