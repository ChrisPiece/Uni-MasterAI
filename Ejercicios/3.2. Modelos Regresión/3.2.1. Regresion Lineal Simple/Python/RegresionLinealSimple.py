#Regresión Lineal Simple
#Importamos las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

#Importamos el dataset
dataset=pd.read_csv('Salary_Data.csv')
dataset

#Extraemos las variables dependientes e independientes.
#La variable independiente es años de experiencia, y la variable dependiente es el salario
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#A continuación dividiremos ambas variables en el conjunto de prueba y el de entrenamiento.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/3, random_state=0)

#Adaptamos nuestro modelo al conjunto de datos de entrenamiento.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Nuestro modelo está listo para predecir resultados.Le pasamos el conjunto.
y_pred=regressor.predict(X_test)

#Visualizamos la predicción de los datos de entrenamiento
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salario VS Experiencia (Datos de entrenamiento)')
plt.xlabel('Años experiencia')
plt.ylabel('Salario')
plt.show()

#Visualizamos la predicción de los datos de entrenamiento
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salario VS Experiencia (Datos de test)')
plt.xlabel('Años experiencia')
plt.ylabel('Salario')
plt.show()