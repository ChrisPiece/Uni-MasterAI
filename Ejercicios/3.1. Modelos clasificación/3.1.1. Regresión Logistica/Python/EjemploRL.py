# Cargamos las librerÃ­as
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# Cargamos el dataset
dataset = pd.read_csv('User_Data.csv')


# Buscamos la relación entre la edad y el salario estimado
# Entradas -> Edad y Salario estimado
x = dataset.iloc[:, [2, 3]].values 
# Salida
y = dataset.iloc[:, 4].values 


# Dividimos el dataset entre 75% entreno y 25% test
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( 
		x, y, test_size = 0.25, random_state = 0) 


# Escalamos la relación entre la edad y el salario estimado.
# Si no hacemos esto, el salario estimado dominará la función y nuestro algoritmo no funcionará adecuadamente.
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain)  
xtest = sc_x.transform(xtest) 
#Ahora nuestros valores están entre -1 y 1.


# Entrenamos nuestro modelo con la funcion LogistisRegression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain)


# Visualizamos el rendimiento del modelo
from matplotlib.colors import ListedColormap 
X_set, y_set = xtest, ytest 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
							stop = X_set[:, 0].max() + 1, step = 0.01), 
					np.arange(start = X_set[:, 1].min() - 1, 
							stop = X_set[:, 1].max() + 1, step = 0.01)) 

plt.contourf(X1, X2, classifier.predict( 
			np.array([X1.ravel(), X2.ravel()]).T).reshape( 
			X1.shape), alpha = 0.75, cmap = ListedColormap(('orange', 'green'))) 

plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 

for i, j in enumerate(np.unique(y_set)): 
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
				c = ListedColormap(('red', 'green'))(i), label = j) 
	

# Mostramos el resultado gráficamente
plt.title('Clasificador (Test)') 
plt.xlabel('Edad') 
plt.ylabel('Salario estimado') 
plt.legend() 
plt.show() 

