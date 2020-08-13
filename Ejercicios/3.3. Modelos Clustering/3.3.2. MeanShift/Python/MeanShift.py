#Ejemplo para Mean Shift con puntos de datos generados aleatoriamente.
 
#Importamos librerías y las funciones a utilizar
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
 
#Creamos los puntos de datos en nuestro gráfico y los ploteamos
np.random.seed(1)
x, _ = make_blobs(n_samples=300, centers=5)
plt.title('Datos aleatorios')
plt.scatter(x[:,0], x[:,1])
plt.show()
 
#Definimos el modelo Mean Shift y ajustamos los datos
mshclust=MeanShift(bandwidth=2.3).fit(x)
 
#Definimos las etiquetas para cada uno de los clusters
labels = mshclust.labels_
centers = mshclust.cluster_centers_
 
#Ploteamos el resultado de los clústeres creados.
plt.scatter(x[:,0], x[:,1], c=labels)
plt.title('Agrupamiento Mean Shift')
plt.scatter(centers[:,0],centers[:,1], marker='+', color="red",s=120 )
plt.show() 
