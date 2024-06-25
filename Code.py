import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the heart disease dataset
df = pd.read_csv('heart.csv')

# Select relevant features for clustering
X = df.drop(columns=['target'])  # Exclude the target column

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def initialize_centers(data, k):
    # Step 1: Apply Principal Component Analysis (PCA) with 2 components
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed_data = pca.transform(data)
    
    # Step 2: Apply percentile for splitting the dataset into K equal parts based on 1st component
    percentiles = np.linspace(0, 100, k + 1)
    split_indices = np.percentile(transformed_data[:, 0], percentiles)
    
    # Step 3: Extract the split datasets from the primary data by index
    split_datasets = []
    for i in range(len(split_indices) - 1):
        start_index = np.where(transformed_data[:, 0] >= split_indices[i])[0][0]
        end_index = np.where(transformed_data[:, 0] < split_indices[i + 1])[0][-1]
        split_datasets.append(data[start_index:end_index + 1])
    
    # Step 4: Calculate the mean of each attribute of the split datasets
    initial_centers = [np.mean(split_dataset, axis=0) for split_dataset in split_datasets]
    
    return np.array(initial_centers)

def assign_clusters(data, centers, clusters, distances):
    
    for i in range(len(data)):
        curr_dist = euclidean_distance(data[i], centers[0])
        if(np.any(curr_dist > distances[i])):
            distances_to_centers = [euclidean_distance(data[i], center) for center in centers]
            nearest_center_index = np.argmin(distances_to_centers)
            clusters[i] = nearest_center_index
            distances[i] = (distances_to_centers[nearest_center_index])
    return np.array(clusters), np.array(distances)

def update_centers(data, clusters, k):
    new_centers = []
    for i in range(k):
        cluster_points = data[clusters == i]
        new_center = np.mean(cluster_points, axis=0)
        new_centers.append(new_center)
    return np.array(new_centers)


def kmeans(data, k, max_iterations=100, tolerance=1e+0, ini_centers=None):

    centers = ini_centers
    num_iterations = 0
    curr_clusters = np.zeros(len(data), dtype = int) 
    curr_dist = np.zeros(len(data), dtype = float) 
    for i in range(max_iterations):
        # Step 2: Assign data points to nearest cluster
        
        #curr_dist = np.full(len(data), np.inf)
        clusters, distances = assign_clusters(data, centers, curr_clusters, curr_dist)
        
        # Step 3: Update cluster centers
        new_centers = update_centers(data, clusters, k)
        
        # Check convergence
        if np.sum(np.abs(new_centers - centers)) < tolerance:
            break
        
        centers = new_centers
        num_iterations += 1
    
    return clusters, distances, centers, num_iterations

# Run K-means clustering
initial_centers = initialize_centers(X.values, k=2)
start_time = time.time()
clusters, distances, centers, num_iterations = kmeans(X.values, k=2, ini_centers=initial_centers)
end_time = time.time()
running_time = end_time - start_time

#print("Number of iterations required for convergence:", num_iterations)
print("Running time:", running_time, "seconds")



# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('K-means Clustering of Heart Disease Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()