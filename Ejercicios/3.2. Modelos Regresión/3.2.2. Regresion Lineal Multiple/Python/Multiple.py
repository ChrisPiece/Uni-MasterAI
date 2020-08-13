#Ejemplo modelo de Regresión Lineal Múltiple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Introducimos los datos manualmente. 
X= [[150,100],[159,200],[170,350],[175,400],[179,500],[180,180],[189,159],[199,110],[199,400],[199,230],[235,120],[239,340],[239,360],[249,145],[249,400]]
Y= [0.73,1.39,2.03,1.45,1.82,1.32,0.83,0.53,1.95,1.27,0.49,1.03,1.24,0.55,1.3]


#Preparamos el dataset.
df2=pd.DataFrame(X,columns=['Precio','GastosPublicidad'])
df2['Ventas']=pd.Series(Y)

#Ajustamos el algoritmo.
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X,Y)

## Preparamos los datos para la visualización.
x_surf, y_surf = np.meshgrid(np.linspace(df2.Precio.min(), df2.Precio.max(), 100),np.linspace(df2.GastosPublicidad.min(), df2.GastosPublicidad.max(), 100))
onlyX = pd.DataFrame({'Precio': x_surf.ravel(), 'GastosPublicidad': y_surf.ravel()})
fittedY=Regressor.predict(onlyX)


#Visualizamos los datos
fig = plt.figure(figsize=(20,10))
### Ajustamos el tamaño de la figura
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Precio'],df2['GastosPublicidad'],df2['Ventas'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Precio')
ax.set_ylabel('Gastos en publicidad')
ax.set_zlabel('Ventas')
plt.show()