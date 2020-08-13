#Instala las librerías una sola vez
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("broom")
#install.packages("ggpubr")

#Importamos las librerías.
library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)

#Recuerda que debes de Importar el dataset desde File ->Import Dataset ->From Text (base)...

#Comprobación de los supuestos de regresión lineal
#No autocorrelación. Al haber solamente una variable de entrada y otra de salida, no es posible que existe alguna relación oculta en los datos
#Normalidad. Los valores son normales
hist(income.data$happiness)

#Linealidad. Vemos que la distribución es mas o menos lineal.
plot(happiness ~ income, data = income.data)

#Procedemos a crear nuestro modelo.

#Visualizamos los puntos de los datos en un gráfico.
income.graph<-ggplot(income.data, aes(x=income, y=happiness))+geom_point()
income.graph

#Dibujamos la línea de regresión
income.graph <- income.graph + geom_smooth(method="lm", col="black")
income.graph

#Dibujamos la ecuación en el gráfico
income.graph <- income.graph +
  stat_regline_equation(label.x = 3, label.y = 7)
income.graph

#Formateamos el gráfico
income.graph +
  theme_bw() +
  labs(title = "Felicidad reportada en función de los ingresos",
       x = "Ingresos (x$10,000)",
       y = "Felicidad puntuación (0 a 10)")
