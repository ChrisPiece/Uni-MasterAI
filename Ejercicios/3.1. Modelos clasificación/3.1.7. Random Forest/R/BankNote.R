#La base de datos sobre la que trabajamos en esta ocasión va sobre falsificación de billetes
#Se digitalizaron 1372 billetes y se determinaron ciertos atributos como la varianza de la imagen, su asimetría, curtosis y entropía.
#La quinta columna nos dice si el billete es falso o no.

# Fuente del dataset, Kaggle: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data#BankNote_Authentication.csv

#Cargamos el dataset, el csv.
#Podemos utilizar la función read.csv o, alternativamente, transformarlo a excel y utilizar read_excel.

banknote <- BankNote_Authentication

#Tenemos que tener en cuenta que los datos tienen su categorización correcta: número, carácter, factor...
#Revisamos y, si es necesario, convertimos la clase (la columna 5) en factor, no 0-1, sino tipo factor para clasificar mejor

banknote$class <- factor(banknote$class)

#Si los otros atributos no estuvieran como número, la función sería parecida:
banknote$variance <- as.numeric(banknote$variance)

#No necesitamos partir los datos en el modelo, sino que tiene pasos automáticos que diciden los grupos de entrenamiento y test.
#Aunque en este caso sí vamos a hacerlo con partición por nuestra cuenta, para que el modelo no tenga que trabajar tanto.

set.seed(999)

#Me quedo con 70% para entrenar el modelo.
#Para elegir ese 70%, primero establezco qué observaciones pertenecerán a ese 70%.
#Así pues, creo una lista de números aleatorios entre 1 y 1372, cuya extensión sea el 70% (es decir, 961 variables)
Datos_entrenamiento <- createDataPartition(banknote$class, p = 0.7, list = F)

#Por lo tanto, la BD para entrenar será la subset con los números de fila aleatorios creados:
DatosParaEntrenar <- subset(banknote[Datos_entrenamiento,])
DatosParaPredecir <- subset(banknote[-Datos_entrenamiento,])

#Y corremos el modelo randomForest
modelRForest <- randomForest(class ~ .,                 #indicamos cuál variable es la dependiente
                             data = DatosParaEntrenar,  #cargamos los datos
                             ntree = 500,               #establecemos el número de árboles
                             keep.forest = TRUE)        #para retener todos los árboles intermedios creados

#Los RF requieren mucha carga computacional, necesita máquinas potentes si hay muchos datos.
#Por eso siempre es bueno reducir el número de árboles al mínimo necesario.
#Graficamos el resultado para vercon cuántos árboles se llega al mínimo error
#Esos árboles son por tanto redundantes y cargan nuestro modelo, vemos ahora el resultado que nos da reducir número de árboles en lo referente a acierto:
#Podemos comparar resultados con menos árboles

modelRForest2 <- randomForest(class ~ .,                 #indicamos cuál variable es la dependiente
                              data = DatosParaEntrenar,  #cargamos los datos
                              ntree = 320,               #establecemos el número de árboles
                              keep.forest = TRUE) 
plot(modelRForest2)
modelRForest2

#Otra forma de conseguir mejor resultado es con más datos. Así pues, si cogieramos todos los datos:

modelRForest3 <- randomForest(class ~ .,                 #indicamos cuál variable es la dependiente
                              data = banknote,           #cargamos todos los datos
                              ntree = 320,               #establecemos el número de árboles
                              keep.forest = TRUE) 

modelRForest3
#Nos fijamos en la tasa de error estimada y la matriz de confusión para comparar
plot(modelRForest3)
#Vemos que necesitaríamos menos árboles para conseguir el mismo nivel de tasa de error.
