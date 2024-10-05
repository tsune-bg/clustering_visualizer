import streamlit as st
import pandas as pd
import os

# タイトルの表示
st.title('CSV ファイルの選択と表示')

# CSV ファイルが格納されているディレクトリのパス
DATASETS_DIR = './datasets'

# ディレクトリが存在するか確認
if not os.path.isdir(DATASETS_DIR):
    st.error(f'ディレクトリ "{DATASETS_DIR}" が存在しません。')
else:
    # ディレクトリ内の CSV ファイルを取得
    csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]

    if not csv_files:
        st.warning('CSV ファイルが見つかりません。')
    else:
        # セレクトボックスでファイルを選択
        selected_file = st.selectbox('表示する CSV ファイルを選択してください。', csv_files)

        # ファイルのパスを作成
        file_path = os.path.join(DATASETS_DIR, selected_file)

        try:
            # CSV を読み込む
            df = pd.read_csv(file_path)

            st.success(f'ファイル "{selected_file}" の読み込みに成功しました。')

            # DataFrame の先頭を表示
            st.subheader('DataFrame の先頭 5 行')
            st.dataframe(df.head())
        except Exception as e:
            st.error(f'ファイルの読み込み中にエラーが発生しました: {e}')
