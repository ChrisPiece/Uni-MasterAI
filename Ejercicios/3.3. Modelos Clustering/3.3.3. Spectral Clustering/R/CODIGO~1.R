install.packages("mlbench")
install.packages("kernlab")

#Creamos el dataset 
library(mlbench)
Datos_iniciales <- mlbench.spirals(100,1,0.025)
Datos <-  4 * Datos_iniciales$x

#Vemos datos iniciales
plot(Datos)

#Librería para Spectral
library(kernlab)

#Aplicamos algoritmo y representamos
sc <- specc(Datos, centers=2)

#Con el gráfico, vemos por colores cómo ha agrupado la tendencia
plot(Datos, col=sc, pch=4)

#Fijamos los grupos iniciales (para posterior comparación)
points(Datos, col=Datos_iniciales$classes, pch=5)

#Ahora contrastamos con cómo sería con Kmeans
cluster_kmeans = kmeans(Datos, 2)

#Y graficamos
plot(Datos, col=cluster_kmeans$cluster, pch=4)   

#Vemos que aquí no agrupa por tendencia, sino por situación física
#Recuperamos los puntos iniciales para contrastar
points(Datos, col=Datos_iniciales$classes, pch=5)

#Vemos cómo los datos que tienen los cuadros y cruces de color contrario, están en el mismo clúster independientemente del método
#Aquellos que están en el centro, que tienen mismo color de cruz y cuadro, son los que entrarían en conflicto