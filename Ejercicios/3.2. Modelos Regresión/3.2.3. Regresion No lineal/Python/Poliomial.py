#Ejemplo de Regresión polinomial o no lineal en Python
#Importamos las librerías 
import matplotlib.pyplot as mtp  
import pandas as pd  
      
#Importamos el dataset  
data_set= pd.read_csv('Position_Salaries.csv')  
  
#Extraemos las variables dependientes e independientes
x= data_set.iloc[:, 1:2].values  
y= data_set.iloc[:, 2].values  

#Ajustamos el dataset al modelo de regresión lineal  
from sklearn.linear_model import LinearRegression  
lin_regs= LinearRegression()  
lin_regs.fit(x,y)

#Ajustamos el dataset al modelo de regresión no lineal  
from sklearn.preprocessing import PolynomialFeatures  
poly_regs= PolynomialFeatures(degree= 3)  
x_poly= poly_regs.fit_transform(x)  
lin_reg_2 =LinearRegression()  
lin_reg_2.fit(x_poly, y)


#Visualizamos los resultados para el modelo de regresión lineal  
mtp.scatter(x,y,color="blue")  
mtp.plot(x,lin_regs.predict(x), color="red")  
mtp.title("Modelo de regresión lineal")  
mtp.xlabel("Categoría profesional")  
mtp.ylabel("Sueldo")  
mtp.show()

#Visualizamos los resultados para el modelo de regresión no lineal 
mtp.scatter(x,y,color="blue")  
mtp.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")  
mtp.title("Modelo de regresión no lineal")  
mtp.xlabel("Categoría profesional")  
mtp.ylabel("Sueldo")  
mtp.show()