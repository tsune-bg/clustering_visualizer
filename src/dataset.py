import numpy as np
import pandas as pd

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

target_dir = 'datasets/'
df1.to_csv(target_dir + 'equal_radius_circles.csv', index=False)
df2.to_csv(target_dir + 'different_radius_circles.csv', index=False)
df3.to_csv(target_dir + 'semicircles_with_noise.csv', index=False)
