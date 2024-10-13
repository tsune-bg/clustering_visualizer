# from src import clustering, visualization
# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('data/equal_radius_circles.csv')
# X = df[['x', 'y']].to_numpy()

# Z = clustering.hierarchical_clustering(X)
# anim = visualization.animation_hierarchical_clustering(X, Z)
# plt.show()

# Import necessary libraries
import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.cluster.hierarchy import dendrogram

# Provided functions for hierarchical clustering and animation
def hierarchical_clustering(X, method='ward'):
    return linkage(X, method=method)

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
        # Clear existing lines (for dendrogram update)
        ax2.cla()
        
        # Get pair of clusters being merged
        cluster_1 = int(Z[i, 0])
        cluster_2 = int(Z[i, 1])

        # Calculate centroid of merged clusters
        new_center = (cluster_centers[cluster_1] + cluster_centers[cluster_2]) / 2.0

        # Update cluster centroid
        cluster_centers[len(X) + i] = new_center

        # Plot line connecting cluster centroids
        line, = ax1.plot(
            [cluster_centers[cluster_1][0], cluster_centers[cluster_2][0]],
            [cluster_centers[cluster_1][1], cluster_centers[cluster_2][1]],
            'r-', lw=2
        )
        lines.append(line)

        # Update dendrogram (showing i+1 step of merging)
        dendrogram(Z, ax=ax2, color_threshold=Z[i, 2], no_labels=True)
        ax2.set_title(f'Dendrogram - Step {i+1}')
    
        return line

    anim = animation.FuncAnimation(fig, update, frames=len(Z), interval=500)
    return anim

# Create a random dataset with 2 features and 50 data points
np.random.seed(42)
X = np.random.rand(50, 2)

# Perform hierarchical clustering
Z = hierarchical_clustering(X, method='ward')

# Final corrected version of the dendrogram color handling logic
def final_animation_hierarchical_clustering_v4(X, Z):
    cluster_members = {i: [X[i]] for i in range(len(X))}  # Track points in each cluster

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_title('Hierarchical Clustering (Ward Method)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    scatter = ax1.scatter(X[:, 0], X[:, 1], s=50, color='gray')

    # Initial dendrogram setup without plotting
    dendrogram(Z, ax=ax2, no_plot=True)
    ax2.set_title('Dendrogram')

    def update(i):
        ax2.cla()  # Clear dendrogram each step

        # Get pair of clusters being merged
        cluster_1 = int(Z[i, 0])
        cluster_2 = int(Z[i, 1])

        # Calculate new cluster membership (merge points of cluster_1 and cluster_2)
        new_cluster_points = cluster_members[cluster_1] + cluster_members[cluster_2]

        # Update cluster membership dictionary
        cluster_members[len(X) + i] = new_cluster_points

        # Clear previous lines and scatter plot
        ax1.cla()
        scatter = ax1.scatter(X[:, 0], X[:, 1], s=50, color='grey')

        # Draw lines connecting all points in the merging clusters
        for point_1 in cluster_members[cluster_1]:
            ax1.scatter(point_1[0], point_1[1], s=50, color='red')
        for point_2 in cluster_members[cluster_2]:
            ax1.scatter(point_2[0], point_2[1], s=50, color='blue')
        ax1.set_title('Hierarchical Clustering (Ward Method)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Update dendrogram:
        def color_func(k):
            # Ensure k is within bounds for Z and check merging distances
            if k < len(X):  # Base clusters should remain gray
                return 'gray'
            elif k == cluster_1:
                return 'red'
            elif k == cluster_2:
                return 'blue'
            elif Z[i, 2] >= Z[k - len(X), 2]:  # If current step distance is larger or equal to the merge
                return 'black'  # Past merges fixed in black
            return ''

        dendrogram(Z, ax=ax2, color_threshold=Z[i, 2], link_color_func=color_func, no_labels=True)
        ax2.set_title(f'Dendrogram - Step {i+1}')

    anim = animation.FuncAnimation(fig, update, frames=len(Z), interval=500)
    return anim

# Generate the corrected animation with accurate dendrogram coloring logic
final_anim_v4 = final_animation_hierarchical_clustering_v4(X, Z)

# Display the corrected animation
plt.show()
# print(X.shape)
# print(Z.shape)
# print(Z)


