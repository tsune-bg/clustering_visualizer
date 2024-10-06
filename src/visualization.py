import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


def plot_dataset(df):
    fig, ax = plt.subplots(1, 1, figsize=[3, 3])
    xlim = (df['x'].min() - 0.1, df['x'].max() + 0.1)
    ylim = (df['y'].min() - 0.1, df['y'].max() + 0.1)
    ax.scatter(df['x'], df['y'], c=df['label'], s=2, cmap='tab10')
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
