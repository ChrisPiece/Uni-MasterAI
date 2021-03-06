#La base de datos sobre la que trabajamos en esta ocasi�n va sobre falsificaci�n de billetes
#Se digitalizaron 1372 billetes y se determinaron ciertos atributos como la varianza de la imagen, su asimetr�a, curtosis y entrop�a.
#La quinta columna nos dice si el billete es falso o no.

# Fuente del dataset, Kaggle: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data#BankNote_Authentication.csv

#Cargamos el dataset, el csv.
#Podemos utilizar la funci�n read.csv o, alternativamente, transformarlo a excel y utilizar read_excel.

banknote <- BankNote_Authentication

#Tenemos que tener en cuenta que los datos tienen su categorizaci�n correcta: n�mero, car�cter, factor...
#Revisamos y, si es necesario, convertimos la clase (la columna 5) en factor, no 0-1, sino tipo factor para clasificar mejor

banknote$class <- factor(banknote$class)

#Si los otros atributos no estuvieran como n�mero, la funci�n ser�a parecida:
banknote$variance <- as.numeric(banknote$variance)

#No necesitamos partir los datos en el modelo, sino que tiene pasos autom�ticos que diciden los grupos de entrenamiento y test.
#Aunque en este caso s� vamos a hacerlo con partici�n por nuestra cuenta, para que el modelo no tenga que trabajar tanto.

set.seed(999)

#Me quedo con 70% para entrenar el modelo.
#Para elegir ese 70%, primero establezco qu� observaciones pertenecer�n a ese 70%.
#As� pues, creo una lista de n�meros aleatorios entre 1 y 1372, cuya extensi�n sea el 70% (es decir, 961 variables)
Datos_entrenamiento <- createDataPartition(banknote$class, p = 0.7, list = F)

#Por lo tanto, la BD para entrenar ser� la subset con los n�meros de fila aleatorios creados:
DatosParaEntrenar <- subset(banknote[Datos_entrenamiento,])
DatosParaPredecir <- subset(banknote[-Datos_entrenamiento,])

#Y corremos el modelo randomForest
modelRForest <- randomForest(class ~ .,                 #indicamos cu�l variable es la dependiente
                             data = DatosParaEntrenar,  #cargamos los datos
                             ntree = 500,               #establecemos el n�mero de �rboles
                             keep.forest = TRUE)        #para retener todos los �rboles intermedios creados

#Los RF requieren mucha carga computacional, necesita m�quinas potentes si hay muchos datos.
#Por eso siempre es bueno reducir el n�mero de �rboles al m�nimo necesario.
#Graficamos el resultado para vercon cu�ntos �rboles se llega al m�nimo error
#Esos �rboles son por tanto redundantes y cargan nuestro modelo, vemos ahora el resultado que nos da reducir n�mero de �rboles en lo referente a acierto:
#Podemos comparar resultados con menos �rboles

modelRForest2 <- randomForest(class ~ .,                 #indicamos cu�l variable es la dependiente
                              data = DatosParaEntrenar,  #cargamos los datos
                              ntree = 320,               #establecemos el n�mero de �rboles
                              keep.forest = TRUE) 
plot(modelRForest2)
modelRForest2

#Otra forma de conseguir mejor resultado es con m�s datos. As� pues, si cogieramos todos los datos:

modelRForest3 <- randomForest(class ~ .,                 #indicamos cu�l variable es la dependiente
                              data = banknote,           #cargamos todos los datos
                              ntree = 320,               #establecemos el n�mero de �rboles
                              keep.forest = TRUE) 

modelRForest3
#Nos fijamos en la tasa de error estimada y la matriz de confusi�n para comparar
plot(modelRForest3)
#Vemos que necesitar�amos menos �rboles para conseguir el mismo nivel de tasa de error.
