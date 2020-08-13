#Ejemplo para Árboles de decisión de regresión

#Importamos librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importamos el dataset y elegimos el nivel del puesto de la persona como variable independiente y el salario como variable
#dependiente a predecir.
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Ajustes la escala de los datos para estandarizarlos y poder predecir los valores de y_pred
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#c_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)

#Ajustamos los datos para el algoritmo que estamso estudiando
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0,max_leaf_nodes=5)
regressor.fit(X,y)
#Si añadimos el parámetro , max_leaf_nodes=5 a nuestro modelo, la salida será diferente

# Una vez que hemos entrenado los datos, intentamos predecir un valor
#x_trans = sc_X.transform([[6.5]])
#y_pred = regressor.predict(x_trans)
#y_pred = sc_y.inverse_transform(y_pred)
#Vemos que la predicción en este caso es 200k, diferente a la salida que nos devolvía en ejercicio de SVR, que era 170k

#Visualizamos el modelo creado
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Ejemplo de Árbol de decisión de Regresión')
plt.xlabel('Nivel Salarial')
plt.ylabel('Salario')
plt.show()