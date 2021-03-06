#Cargamos la base de datos iris
data("iris")

#Y las librer�as correspondientes
library(ggplot2)

#Hacemos una previsualizaci�n de los datos usando dos variables, longitud y anchura del p�talo
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Vemos c�mo son los datos
summary(iris$Species)

#Tenemos informaci�n sobre los datos. Sin embargo, la mayor�a de veces que usemos clustering no tendremos tan claro esos atributos de clasificaci�n como es, en este caso, por especies

#Indicamos que, de la base de datos iris, s�lo cogemos la columna 3 y 4
#Pasamos a dividir en cluster. Hacemos en este caso s�lo 2:
cluster = kmeans(iris[,3:4], 2)
cluster

#Agregamos la variable cluster (cluster 1 o cluster 2) en iris:
iris$cluster <- cluster$cluster
iris$cluster <- as.factor(iris$cluster)

#Hacemos un gr�fico indicando que el cluster lo determine el color:
ggplot(iris, aes(Petal.Length, Petal.Width, color = cluster)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Hacemos una especie de matriz de confusi�n para ver c�mo se ha agrupado:
tabla <- table(iris$cluster,
               iris$Species)
tabla

#Vemos en este caso que la setosa est� perfectamente definida en un cluster (2)
#Tambi�n lo est� virginica (1). Sin embargo, versicolor, que es muy similar en esos dos atributos a virginica, se le ha ido una observaci�n (la de menor longtud de p�talo) al cluster 2.

#Pasamos a crear una funci�n para el punto de codo.
#Esta funci�n se aplicar� sobre un dataframe (df) nuevo que crearemos con s�lo los atributos que vamos a estudiar.
#En esta funci�n, se establece un n�mero k de 1 a 10.
#El error se basa en la suma del cuadrado de todos los errores dentro de un cluster.
#Este error debe ser peque�o, lo que significar� que la varianza dentro del grupo es peque�a y el cluster est� bien identificado.

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

#Una vez tenemos la funci�n, establecemos que el df es la columna 3 y 4
df = iris[,3:4]

#Y creamos el objeto de la funci�n aplicada a esos datos
Elbow <- elbowfun(df)  

#Para ver el resultado visual, graficamos ese objeto
ggplot(Elbow, aes(Num_k, Error))  + geom_line() +
  geom_point(color = "blue", size = 3) + xlab("Num_k") + ylab ("Error")

#Vemos c�mo el error baja dr�sticamente con dos cl�steres, y sigue bajando con 3 (como podr�amos deducir por los datos que tenemos)

#Ahora bien, �por qu� los siguientes tienen un error similar? (explicaci�n matem�tica)
#Pues porque la diferencia entre los grupos (varianza, el error) es m�nima y, por tanto, hacer m�s grupos no conlleva un menor error.