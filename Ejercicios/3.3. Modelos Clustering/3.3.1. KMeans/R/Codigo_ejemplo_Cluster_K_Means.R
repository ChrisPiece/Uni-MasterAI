#Cargamos la base de datos iris
data("iris")

#Y las librerías correspondientes
library(ggplot2)

#Hacemos una previsualización de los datos usando dos variables, longitud y anchura del pétalo
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Vemos cómo son los datos
summary(iris$Species)

#Tenemos información sobre los datos. Sin embargo, la mayoría de veces que usemos clustering no tendremos tan claro esos atributos de clasificación como es, en este caso, por especies

#Indicamos que, de la base de datos iris, sólo cogemos la columna 3 y 4
#Pasamos a dividir en cluster. Hacemos en este caso sólo 2:
cluster = kmeans(iris[,3:4], 2)
cluster

#Agregamos la variable cluster (cluster 1 o cluster 2) en iris:
iris$cluster <- cluster$cluster
iris$cluster <- as.factor(iris$cluster)

#Hacemos un gráfico indicando que el cluster lo determine el color:
ggplot(iris, aes(Petal.Length, Petal.Width, color = cluster)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Hacemos una especie de matriz de confusión para ver cómo se ha agrupado:
tabla <- table(iris$cluster,
               iris$Species)
tabla

#Vemos en este caso que la setosa está perfectamente definida en un cluster (2)
#También lo está virginica (1). Sin embargo, versicolor, que es muy similar en esos dos atributos a virginica, se le ha ido una observación (la de menor longtud de pétalo) al cluster 2.

#Pasamos a crear una función para el punto de codo.
#Esta función se aplicará sobre un dataframe (df) nuevo que crearemos con sólo los atributos que vamos a estudiar.
#En esta función, se establece un número k de 1 a 10.
#El error se basa en la suma del cuadrado de todos los errores dentro de un cluster.
#Este error debe ser pequeño, lo que significará que la varianza dentro del grupo es pequeña y el cluster está bien identificado.

elbowfun <- function(df) {
  Num_k <- rep(NA,10)
  Error <- rep(NA,10)
  for (i in 1:10) {
    cluster <- kmeans(df, i)
    Num_k[i] <- i
    Error[i] <- cluster$tot.withinss
  }
  df_nuevo <- data.frame(Num_k,Error)
  return (df_nuevo)
}

#Una vez tenemos la función, establecemos que el df es la columna 3 y 4
df = iris[,3:4]

#Y creamos el objeto de la función aplicada a esos datos
Elbow <- elbowfun(df)  

#Para ver el resultado visual, graficamos ese objeto
ggplot(Elbow, aes(Num_k, Error))  + geom_line() +
  geom_point(color = "blue", size = 3) + xlab("Num_k") + ylab ("Error")

#Vemos cómo el error baja drásticamente con dos clústeres, y sigue bajando con 3 (como podríamos deducir por los datos que tenemos)

#Ahora bien, ¿por qué los siguientes tienen un error similar? (explicación matemática)
#Pues porque la diferencia entre los grupos (varianza, el error) es mínima y, por tanto, hacer más grupos no conlleva un menor error.