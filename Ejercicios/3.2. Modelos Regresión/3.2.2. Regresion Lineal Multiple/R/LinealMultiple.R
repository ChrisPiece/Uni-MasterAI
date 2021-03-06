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

#Mostramos un resumen de los datos
summary(heart.data)

#Comprobamos si los supuestos de la regresi�n lineal son correctos.

#No existe una correlaci�n entre ls variables independientes.
cor(heart.data$biking, heart.data$smoking)

#La variable dependiente tiene una distribuci�n normal.
hist(heart.data$heart.disease)

#Los datos mantienen cierta linealidad.
plot(heart.disease ~ biking, data=heart.data)
plot(heart.disease ~ smoking, data=heart.data)

#Testeamos la linealidad de todas las variables. Los valores que nos devuelven son muy bajos, por lo que podemos continuar.
heart.disease.lm<-lm(heart.disease ~ biking + smoking, data = heart.data)

summary(heart.disease.lm)

#Comprobamos si nuestro modelo creado se ajusta bien a los datos. Los valores residuales no muestran sesgo alguno.
par(mfrow=c(2,2))
plot(heart.disease.lm)
par(mfrow=c(1,1))


#Al tener diversas variables independientes es mas complicado mostrar el resultado gr�ficamente.
#Para este ejemplo vamos a mostrar la relaci�n entre el ciclismo y la enfermedad card�aca en diferentes niveles de tabaquismo.

plotting.data<-expand.grid(
  biking = seq(min(heart.data$biking), max(heart.data$biking), length.out=30),
  smoking=c(min(heart.data$smoking), mean(heart.data$smoking), max(heart.data$smoking)))

plotting.data$predicted.y <- predict.lm(heart.disease.lm, newdata=plotting.data)

plotting.data$smoking <- round(plotting.data$smoking, digits = 2)

plotting.data$smoking <- as.factor(plotting.data$smoking)

#Ploteamos los datos originales
heart.plot <- ggplot(heart.data, aes(x=biking, y=heart.disease)) +
  geom_point()

heart.plot

#A�adimos las l�neas de regresi�n
heart.plot <- heart.plot +
  geom_line(data=plotting.data, aes(x=biking, y=predicted.y, color=smoking), size=1.25)

heart.plot

#Asignamos t�tulos a nuestro resultado.
heart.plot <-
  heart.plot +
  theme_bw() +
  labs(title = "Tasas de enfermedad card�aca (% de poblaci�n) \n en funci�n de ir al trabajo en bicicleta y fumar",
       x = "Ir en bicicleta al trabajo (% de poblaci�n)",
       y = "Enfermedad card�aca (% de poblaci�n)",
       color = "Fumar \n (% de poblaci�n)")

heart.plot

#Vemos que la enfermedad card�aca aumenta si la persona fuma y no va al trabajo en bicicleta.

