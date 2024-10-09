from src import clustering, visualization
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/equal_radius_circles.csv')
X = df[['x', 'y']].to_numpy()

Z = clustering.hierarchical_clustering(X)
anim = visualization.animation_hierarchical_clustering(X, Z)
plt.show()