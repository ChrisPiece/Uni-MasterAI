#Ejemplo para Análisis Discriminante Lineal

#Importamos las librerías necesarias.
from sklearn import datasets
import matplotlib.pyplot as plt

#Cargamos el dataset Iris
iris = datasets.load_iris()

#Extraemos las variables deependientes y las variables independientes.
X = iris.data
y = iris.target

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Obtenemos los características de las variables
target_names = iris.target_names

#Ajustamos los datos al modelo de ADL con una reducción de dimensionalidad de 2
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

#Asignamos los colores para los puntos
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)

#Dibujamos el gráfico
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('ADL del dataset Iris')

plt.show()