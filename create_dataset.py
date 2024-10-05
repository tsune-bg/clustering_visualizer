import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to generate uniformly distributed points in a circle with correct density
def generate_uniform_circle_data(radius, center, num_points):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, 1, num_points)) * radius  # Correct density for uniform distribution
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

# Function to generate Gaussian distributed points along a specific angular range of a semicircle
def generate_semicircle_data_fixed_range(radius, center, num_points, angle_range, std_dev=0.1):
    angles = np.random.uniform(angle_range[0], angle_range[1], num_points)
    x = center[0] + radius * np.cos(angles) + np.random.normal(0, std_dev, num_points)
    y = center[1] + radius * np.sin(angles) + np.random.normal(0, std_dev, num_points)
    return np.column_stack((x, y))

# Step 1: Generate the corrected three datasets

# Dataset 1: Two non-overlapping circles with equal radii, uniformly distributed
data1_class1 = generate_uniform_circle_data(1, [-1.1, 0], 500)
data1_class2 = generate_uniform_circle_data(1, [1.1, 0], 500)
data1 = np.vstack((data1_class1, data1_class2))

# Dataset 2: Two non-overlapping circles with different radii, uniformly distributed
data2_class1 = generate_uniform_circle_data(1, [-1.1, 0], 500)
data2_class2 = generate_uniform_circle_data(3, [3.3, 0], 500)  # Corrected radius difference
data2 = np.vstack((data2_class1, data2_class2))

# Dataset 3: Two semicircles with Gaussian noise, different angle ranges and radius-based shift
data3_class1 = generate_semicircle_data_fixed_range(1, [0, 0], 500, [0, np.pi])
data3_class2 = generate_semicircle_data_fixed_range(1, [1, 0], 500, [np.pi, 2 * np.pi])  # Shifted by radius, angle range adjusted
data3 = np.vstack((data3_class1, data3_class2))

# Convert datasets to pandas DataFrame for ease of use
df1 = pd.DataFrame(data1, columns=['x', 'y'])
df2 = pd.DataFrame(data2, columns=['x', 'y'])
df3 = pd.DataFrame(data3, columns=['x', 'y'])

df1['label'] = [0] * 500 + [1] * 500
df2['label'] = [0] * 500 + [1] * 500
df3['label'] = [0] * 500 + [1] * 500


# Visualize the corrected three datasets
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Dataset 1: Equal Radius Non-overlapping Circles with Labels
ax[0].scatter(df1['x'], df1['y'], c=df1['label'], cmap='tab10', label='Dataset 1')
ax[0].set_title('Dataset 1: Equal Radius Circles (Labeled)')
ax[0].set_aspect('equal')

# Dataset 2: Different Radius Non-overlapping Circles with Labels
ax[1].scatter(df2['x'], df2['y'], c=df2['label'], cmap='tab10', label='Dataset 2')
ax[1].set_title('Dataset 2: Different Radius Circles (Labeled)')
ax[1].set_aspect('equal')

# Dataset 3: Semicircles with Gaussian Noise with Labels
ax[2].scatter(df3['x'], df3['y'], c=df3['label'], cmap='tab10', label='Dataset 3')
ax[2].set_title('Dataset 3: Semicircles with Gaussian Noise (Labeled)')
ax[2].set_aspect('equal')

plt.show()