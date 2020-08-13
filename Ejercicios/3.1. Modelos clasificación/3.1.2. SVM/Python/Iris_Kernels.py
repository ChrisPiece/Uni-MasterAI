#Ejemplos modelo de clasificación Máquinas de soporte de Vector

# Importamos las librerías y módulos
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Importamos los datos
iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target

h = .02

# Ajustamos el parámetro de regularización
C = 1.0 

#Aplicamos los parámetros a nuestros datos y ajustamos el modelo.
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# Creamos los marcos para las gráficas
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Colocamos títulos para las gráficas
titles = ['SVM con kernel lineal',
 'Lineal SVM',
 'SVM con RBF kernel',
 'SVM con polinomio(grado 3) kernel']


for i, clf in enumerate((svc,lin_svc, rbf_svc, poly_svc)):
 # Establecemos límites para las fronteras  de los gráficos
 plt.subplot(2, 2, i + 1)
 plt.subplots_adjust(wspace=0.4, hspace=0.4)

 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

 # Coloreamos las gráficas
 Z = Z.reshape(xx.shape)
 plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

 # Entrenamiento del modelo
 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
 plt.xlabel('Longitud Sepal')
 plt.ylabel('Peso Sepal')
 plt.xlim(xx.min(), xx.max())
 plt.ylim(yy.min(), yy.max())
 plt.xticks(())
 plt.yticks(())
 plt.title(titles[i])

plt.show()