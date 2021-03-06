#Ejemplo An�lisis Discriminante Lineal con el dataset Iris para R

# Cargamos el dataset
data(iris)
datos <- iris
View(datos)

#Dividimos los datos entre entrenamiento y test
set.seed(101)
tamano.total <- nrow(datos)
tamano.entreno <- round(105*0.7)
datos.indices <- sample(1:tamano.total , size=tamano.entreno)
datos.entreno <- datos[datos.indices,]
datos.test <- datos[-datos.indices,]

# Cargamos las librer�as
require(MASS) 

#Ajustamos los datos al modelo de An�lisis Discriminante Lineal
datos.entreno.lda <- lda(formula= Species~. , data=datos.entreno)

# Asignamos colores para cada clase
color <- rep("green",nrow(datos.entreno))
color[datos.entreno$Species == "setosa"] <- "red"
color[datos.entreno$Species == "virginica"] <- "blue"

#Seleccionamos la dimensi�n y mostramos el gr�fico
plot(datos.entreno.lda, dimen=2, col=color, abbrev=3)
