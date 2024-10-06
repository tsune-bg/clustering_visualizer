from src import clustering, visualization
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/equal_radius_circles.csv')
X = df[['x', 'y']].to_numpy()

center_history, label_history = clustering.kmeans(X, 2, 23)
ani = visualization.animation_kmeans(X, center_history, label_history)
plt.show()