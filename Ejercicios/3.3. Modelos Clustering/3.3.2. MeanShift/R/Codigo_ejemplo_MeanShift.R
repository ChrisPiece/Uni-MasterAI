#Instalamos paquetes necesarios
install.packages('meanShiftR')
install.packages("ggplot2")

#Cargamos datos iris
data("iris")

#Representamos los datos iniciales, para tener una idea
library(ggplot2)
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Cogemos info sólo del pétalo
Petalos  <- cbind(iris$Petal.Length, iris$Petal.Width)

#Aplicamos algoritmo
library(meanShiftR)
Etiquetas <- meanShift(queryData = Petalos, trainData = Petalos, bandwidth = c(0.7,0.7))

#Añadimos resultados del algoritmo a la BD inicial
iris$Etiquetas <- as.factor(Etiquetas$assignment)

#Representamos la predicción del modelo
ggplot(iris, aes(Petal.Length, Petal.Width, color = Etiquetas)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo")

#Matriz de confusión
MC <- table(iris$Etiquetas,
            iris$Species)
MC

#Precisión del modelo
sum(diag(MC))/sum(MC)

#Creamos y graficamos los centroides
#Utilizamos la columna value de las etiquetas
Centroides <- unique(Etiquetas$value)

ggplot(iris, aes(Petal.Length, Petal.Width, color = Etiquetas)) + 
  geom_point() + xlab("Longitud petalo") + ylab ("Anchura petalo") +
  geom_point(aes(x= Centroides[1,1], y= Centroides[1,2]), colour="red",shape = 24)+
  geom_point(aes(x= Centroides[2,1], y= Centroides[2,2]), colour="green",shape = 24)+
  geom_point(aes(x= Centroides[3,1], y= Centroides[3,2]), colour="blue",shape = 24)
