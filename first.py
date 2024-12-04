import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


file_path = 'Mall_Customers.csv' 
data = pd.read_csv(file_path)


X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


wcss = []  

for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-cluster Sum of Squares)')
plt.xticks(range(1, 11))
plt.grid()
plt.show()


optimal_clusters = 5 
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_clusters):
    plt.scatter(
        X.values[y_kmeans == i, 0], 
        X.values[y_kmeans == i, 1], 
        s=100, 
        c=colors[i], 
        label=f'Cluster {i+1}'
    )

plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    s=300, 
    c='yellow', 
    marker='*', 
    label='Centroids'
)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
