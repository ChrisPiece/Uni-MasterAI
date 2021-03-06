#Instala las librer�as una sola vez
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("broom")
#install.packages("ggpubr")

#Importamos las librer�as.
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)

#Recuerda que debes de Importar el dataset desde File ->Import Dataset ->From Text (base)...

#Comprobaci�n de los supuestos de regresi�n lineal
#No autocorrelaci�n. Al haber solamente una variable de entrada y otra de salida, no es posible que existe alguna relaci�n oculta en los datos
#Normalidad. Los valores son normales
hist(income.data$happiness)

#Linealidad. Vemos que la distribuci�n es mas o menos lineal.
plot(happiness ~ income, data = income.data)

#Procedemos a crear nuestro modelo.

#Visualizamos los puntos de los datos en un gr�fico.
income.graph<-ggplot(income.data, aes(x=income, y=happiness))+geom_point()
income.graph

#Dibujamos la l�nea de regresi�n
income.graph <- income.graph + geom_smooth(method="lm", col="black")
income.graph

#Dibujamos la ecuaci�n en el gr�fico
income.graph <- income.graph +
  stat_regline_equation(label.x = 3, label.y = 7)
income.graph

#Formateamos el gr�fico
income.graph +
  theme_bw() +
  labs(title = "Felicidad reportada en funci�n de los ingresos",
       x = "Ingresos (x$10,000)",
       y = "Felicidad puntuaci�n (0 a 10)")
