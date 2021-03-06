#Instalamos el paquete necesario para hacer SVM en R
install.packages("e1071")

#Si lo tuvi�semos instalado, ejecutarlo
library("e1071")

#Cargamos la base de datos "Cats". Como Iris, es una base de datos disponible en R para entrenamiento.
#Esta base de datos nos da una lista de 144 registros con el sexo del gato, su peso y el tama�o de su coraz�n. 
data(cats, package = 'MASS')

#Creamos un conjunto de entrenamiento y test.
#Con la funci�n sample dividimos la muestra en N grupos (en este caso, 2)
#Asignamos el vector de probabilidad para entrenamiento. 60% para test y 40% para entrenamiento
ind <- sample(2, nrow(cats), replace=TRUE, prob=c(0.6, 0.4))

#El grupo de test, que son aquellos casos que han tomado el valor 1 en la divisi�n anterior
grupo_test <- cats[ind==1,]
#El grupo de entrenamiento, que son aquellos casos que han tomado el valor 2 en la divisi�n anterior
grupo_entrenamiento <- cats[ind==2,]

#Entrenamos el modelo
#Aplicamos la funci�n svm.
#La variable dependiente va a ser Sexo y las otras dos las variables dependientes
#Los datos que se usan para entrenar el modelo son el trainset.
modelo <- svm(Sex~., data=grupo_entrenamiento, kernel="radial")
modelo

#Cuando la separaci�n es sencilla, de dos categor�as, ser�a lineal.
#Sin embargo, la mayor�a de las veces es algo m�s complejo y hay que utilizar el m�todo kernel adecuado.
#Por defecto viene el radial. M�s informaci�n en la consola: "?svm"

#Posteriormente, le aplicamos la parte de predicc�n a los datos de test (el 20% que nos quedeaba).
prediccion <- predict(modelo, newdata=grupo_test[-1])

#Hacemos un gr�fico para ver c�mo ser�a la separaci�n del modelo
plot(modelo, cats)

#El "Number of Support Vectors" nos dir�a qu� cantidad de vectores han sido necesarios para establecer esa separaci�n entre los grupos
#Si con un solo hiperplano, hubi�ramos podido establecer la clasificaci�n, este valor ser�a 1.

#Pasamos a crear la matriz de confusi�n, para ver la fiabilidad del modelo
MC <- table(grupo_test[,1], prediccion)
#Lo visualizamos:
MC

#En la diagonal de izquierda a derecha y de arriba hacia abajo, vemos los casos que se han identificado correctamente
#Para calcular la tasa de acierto, sumamos las diagonales y dividimos entre el total.
acierto <- (sum(diag(MC)))/sum(MC)
acierto
