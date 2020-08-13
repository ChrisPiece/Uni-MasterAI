#BACKGROUND

install.packages('naivebayes')
#Volvemos a utilizar la base de datos ya conocida Iris
data('iris')

#PREGUNTA 1: Establece semilla aleatoria y, con la función sample (paquete base), que haga también remplazo de 1:150. 
#SOLUCIÓN 1:
set.seed(99) 
rnum<- sample(rep(1:150)) 
rnum

#Tenemos que:
iris<- iris[rnum,] 

#PREGUNTA 2: Establece un grupo de entrenamiento del 1:130 y que el target (el objetivo, lo que queremos acertar) es la columna 5 de cada uno de esos 130.
#PREGUNTA 2: Establece igualmente un grupo de test del 131 al 150 y replica el target como en el apartado anterior.
#SOLUCIÓN 2:
iris.train<- iris[1:130,]
iris.train.target<- iris[1:130,5]
iris.test<- iris[131:150,]
iris.test.target<- iris[131:150,5]

#Tenemos que:
library(naivebayes)
modelo <- naive_bayes(formula = Species ~ .,  data = iris.train)

#PREGUNTA 3: Crea una predicción del modelo con los datos test.
#SOLUCIÓN 3:
predicciones <- predict(modelo, iris.test)
tabla <- table(iris.test.target, predicciones)
print(tabla)

#PREGUNTA 4: Con la matriz resultante en la tabla anterior, calcula el acierto del modelo.
#SOLUCIÓN 4: 
sum(diag(tabla))/sum(tabla)

