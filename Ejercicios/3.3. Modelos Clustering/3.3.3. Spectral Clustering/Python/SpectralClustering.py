#Ejemplo para Spectral Clustering con puntos de datos generados aleatoriamente en forma de media luna.

#Importamos las librerías necesarias para nuestro ejemplo.
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

#Generamos los datos aleatoriamente en forma de media luna y los mostramos en gráfico
random_state = 21
X, y = make_moons(300, noise=.08)
fig, ax = plt.subplots(figsize=(9,5))
ax.set_title('Datos tipo media luna')
ax.scatter(X[:, 0], X[:, 1],s=30)
plt.show()

#Definimos el modelo Spectral Clustering. Número de clúster y el tipo de afinidad, en este caso KNN
spectral = SpectralClustering(
    n_clusters=2,
    affinity='nearest_neighbors',
    n_neighbors=15,
    assign_labels='kmeans')

#Ajustamos los datos al modelo.
etiquetas = spectral.fit_predict(X)

#Mostramos los datos agrupados con Spectral Clustering
fig, ax = plt.subplots(figsize=(9,5))
ax.set_title('Spectral Clustering')
plt.scatter(X[:, 0], X[:, 1], c=etiquetas, s=40, cmap='plasma')
plt.show()

#Mostramos los datos agrupados con Spectral Clustering
from sklearn.cluster import KMeans
Clusters = KMeans(n_clusters=2).fit(X)
etiquetas_clust = Clusters.predict(X)
plt.title('K - Means')
plt.scatter(X[:, 0], X[:, 1], c=etiquetas_clust, s=40, cmap='plasma')
plt.show()