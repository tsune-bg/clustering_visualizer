import numpy as np
from scipy.cluster.hierarchy import linkage


def kmeans(X, k, random_state=42):
    np.random.seed(random_state)

    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    prev_centers = np.zeros(centers.shape)
    labels = np.zeros(X.shape[0], dtype=np.int64)
    center_history = [centers.copy()]
    label_history = []

    while not np.allclose(centers, prev_centers):
        # Step 1: Select the closest center point to each point
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centers, axis=1)
            labels[i] = np.argmin(distances)
        print(len(labels[labels == 0]))
        label_history.append(labels.copy())

        # Step 2: Update center points
        prev_centers = centers.copy()
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                centers[j] = np.mean(points, axis=0)
        center_history.append(centers.copy())

    return center_history, label_history

def hierarchical_clustering(X, method='ward'):
    return linkage(X, method=method)
