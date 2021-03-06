install.packages("mlbench")
install.packages("kernlab")

#Creamos el dataset 
library(mlbench)
Datos_iniciales <- mlbench.spirals(100,1,0.025)
Datos <-  4 * Datos_iniciales$x

#Vemos datos iniciales
plot(Datos)

#Librer�a para Spectral
library(kernlab)

#Aplicamos algoritmo y representamos
sc <- specc(Datos, centers=2)

#Con el gr�fico, vemos por colores c�mo ha agrupado la tendencia
plot(Datos, col=sc, pch=4)

#Fijamos los grupos iniciales (para posterior comparaci�n)
points(Datos, col=Datos_iniciales$classes, pch=5)

#Ahora contrastamos con c�mo ser�a con Kmeans
cluster_kmeans = kmeans(Datos, 2)

#Y graficamos
plot(Datos, col=cluster_kmeans$cluster, pch=4)   

#Vemos que aqu� no agrupa por tendencia, sino por situaci�n f�sica
#Recuperamos los puntos iniciales para contrastar
points(Datos, col=Datos_iniciales$classes, pch=5)

#Vemos c�mo los datos que tienen los cuadros y cruces de color contrario, est�n en el mismo cl�ster independientemente del m�todo
#Aquellos que est�n en el centro, que tienen mismo color de cruz y cuadro, son los que entrar�an en conflicto