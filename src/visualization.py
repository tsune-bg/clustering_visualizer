import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
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
