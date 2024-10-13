import numpy as np
import plotly.express as px
import plotly.graph_objs as go


colors = px.colors.qualitative.Plotly


def plot_dataset(df):
    # Plotly Expressで散布図を作成
    df['label'] = df['label'].astype('category')
    fig = px.scatter(df, x='x', y='y', color='label',
                     size_max=10, opacity=0.6,
                     height=300, color_discrete_sequence=colors)

    # レイアウト調整
    fig.update_layout(
        xaxis=dict(ticks='', showticklabels=False, scaleanchor='y'),
        yaxis=dict(ticks='', showticklabels=False),
        showlegend=False
    )

    return fig

def animation_kmeans(X, center_history, label_history):
    # 初期プロット
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(ticks='', showticklabels=False, scaleanchor='y'),
            yaxis=dict(ticks='', showticklabels=False),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                # "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0,
                "xanchor": "left",
                "y": -0.5,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Iteration: "},
                "steps": [
                    {"args": [[str(i)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                     "label": f"Iteration {i + 1}",
                     "method": "animate"} for i in range(len(label_history))
                ]
            }]
        )
    )

    # フレームを作成
    frames = []
    for step in range(len(label_history)):
        frame_data = []
        
        # クラスタのラベルごとに点を追加
        unique_labels = np.unique(label_history[step])
        for i in unique_labels:
            frame_data.append(go.Scatter(
                x=X[label_history[step] == i, 0],
                y=X[label_history[step] == i, 1],
                mode='markers',
                marker=dict(size=10, opacity=0.6, color=colors[i]),
                name=f'Cluster {i + 1}'
            ))

        # セントロイド（クラスタ中心）の追加
        frame_data.append(go.Scatter(
            x=center_history[step][:, 0],
            y=center_history[step][:, 1],
            mode='markers',
            marker=dict(symbol='x', size=12, color='black'),
            name='Centroids'
        ))

        # フレーム追加
        frames.append(go.Frame(data=frame_data, name=str(step)))

    # フレームを追加
    fig.frames = frames

    # 初期状態で最初のフレームを表示
    initial_frame = frames[0]['data']
    fig.add_traces(initial_frame)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig