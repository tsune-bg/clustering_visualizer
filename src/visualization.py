import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


sns.set_theme()


def plot_dataset(df):
    xlim = (df['x'].min() - 0.1, df['x'].max() + 0.1)
    ylim = (df['y'].min() - 0.1, df['y'].max() + 0.1)
    height = 1
    fig, ax = plt.subplots(figsize=(height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]), height))
    # fig, ax = plt.subplots(figsize=[3, 3])

    ax.scatter(df['x'], df['y'], c=df['label'], s=1, cmap='tab10')
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def animation_kmeans(X, center_history, label_history):
    xlim = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    ylim = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    height = 4
    fig, ax = plt.subplots(figsize=(height * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]), height))

    def update(step):
        ax.clear()
        ax.set_title(f'Iteration {step + 1}')

        # Plot points based on current labels
        unique_labels = np.unique(label_history[step])
        for i in unique_labels:
            ax.scatter(
                X[label_history[step] == i, 0],
                X[label_history[step] == i, 1],
                label=f'Cluster {i + 1}',
                alpha=0.6
            )

        # Plot center points
        ax.scatter(
            center_history[step][:, 0],
            center_history[step][:, 1],
            c='black',
            marker='x',
            s=30,
            label='Centroids'
        )

        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
        ax.set_ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)
        ax.set_xticks([])
        ax.set_yticks([])

    anim = animation.FuncAnimation(fig, update, frames=len(label_history), interval=750)
    return anim

def animation_hierarchical_clustering(X, Z):
    cluster_centers = {i: X[i] for i in range(len(X))}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_title('Hierarchical Clustering (Ward Method)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax1.scatter(X[:, 0], X[:, 1], s=1)
    dendrogram(Z, ax=ax2, no_plot=True)
    ax2.set_title('Dendrogram')

    lines = []
    def update(i):
        # 既存の線をクリア（デンドログラム更新用）
        ax2.cla()
        
        # 各リンクのペアを取得
        cluster_1 = int(Z[i, 0])
        cluster_2 = int(Z[i, 1])

        # 結合されたクラスタの重心を計算
        new_center = (cluster_centers[cluster_1] + cluster_centers[cluster_2]) / 2.0

        # クラスタの重心を更新
        cluster_centers[len(X) + i] = new_center

        # 各クラスタの点を結ぶ線を描画
        line, = ax1.plot(
            [cluster_centers[cluster_1][0], cluster_centers[cluster_2][0]],
            [cluster_centers[cluster_1][1], cluster_centers[cluster_2][1]],
            'r-', lw=2
        )
        lines.append(line)

        # デンドログラムの更新（i+1回目の結合を表示）
        dendrogram(Z, ax=ax2, color_threshold=Z[i, 2], no_labels=True)
        # dendrogram(Z, ax=ax2, color_threshold=Z[i, 2], link_color_func=lambda k: link_colors[k], no_labels=True)
        ax2.set_title(f'Dendrogram - Step {i+1}')
    
        return line

    anim = animation.FuncAnimation(fig, update, frames=len(Z), interval=10)
    return anim

