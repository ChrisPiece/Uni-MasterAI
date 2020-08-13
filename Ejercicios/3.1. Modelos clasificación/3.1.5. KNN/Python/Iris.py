#Ejemplo de Vecinos mas cercanos para clasificacion

#Importamos las librerias
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#Establecemos una variable para el número k
n_neighbors = 5

# Importamos los datos
iris = datasets.load_iris()

# Usamos únicamente las coordenadas 2D
X = iris.data[:, :2]
y = iris.target

# Definimos el tamaño del paso en la malla
h = .02

# Creamos los mapas de color
cmap_light = ListedColormap(['#FF1AAA', '#AAFAA1', '#AAAAA1'])
cmap_bold = ListedColormap(['#FF0000', '#19FA05', '#AAAA99'])

for weights in ['uniform', 'distance']:
# creamos una instancia del Clasificador de Vecinos y ajustamos los datos al algoritmo importado de sklearn.
# Todo el peso recae aquí, posteriormente dibujaremos losresultados.
  clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
  clf.fit(X, y)

#Plotea el límite de decisión. Para ello, asignaremos un color a cada uno de ellos.
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

  # Coloreamos el resultado del algoritmo
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

  # Coloremos tambien los puntos de entrenamiento
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.title("3-Class classification (k = %i, weights = '%s')"
            % (n_neighbors, weights))

plt.show()