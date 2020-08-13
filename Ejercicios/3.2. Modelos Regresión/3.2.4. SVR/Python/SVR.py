#Ejemplo para Máquinas de Soporte de Vectores de Regresión

#Importamos las librerías que vamos a usar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importamos el dataset y elegimos el nivel del puesto de la persona como variable independiente y el salario como variable
#dependiente a predecir.
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Ajustes la escala de los datos para estandarizarlos y poder tratar con ellos
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Utilizando la función de Kernel RBF ajustamos los datos
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Una vez que hemos entrenado los datos, intentamos predecir un valor
x_trans = sc_X.transform([[6.5]])
y_pred = regressor.predict(x_trans)
y_pred = sc_y.inverse_transform(y_pred)
#El valor que nos devuelve es 170k, por lo que vemos que es correcto.


#Visualizamos gráficamente los valores reales
x_real = sc_X.inverse_transform(X)
y_real = sc_y.inverse_transform(y)

X_grid = np.arange(min(x_real), max(x_real), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))

x_grid_transform = sc_X.transform(X_grid)

y_grid = regressor.predict(x_grid_transform)
y_grid_real = sc_y.inverse_transform(y_grid)

plt.scatter(x_real, y_real, color = 'red')
plt.plot(X_grid, y_grid_real, color = 'blue')
plt.title('Ejemplo de SVR con Kernel RBF con epsilon=0.1 y C=1.0')
plt.xlabel('Nivel Salarial')
plt.ylabel('Salario')
plt.show()