# Usamos las siguientes librerías para manejar y visualizar datos
library(tidyverse)
library(caret)
theme_set(theme_bw())


# Cargamos los datos
data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
# Inspecionamos los datos
sample_n(PimaIndiansDiabetes2, 3)
# Dividimos los datos entre entrenamiento y test
set.seed(123)
training.samples <- PimaIndiansDiabetes2$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- PimaIndiansDiabetes2[training.samples, ]
test.data <- PimaIndiansDiabetes2[-training.samples, ]


# Ajustamos el modelo
model <- glm( diabetes ~., data = train.data, family = binomial)
# Resumimos el modelo
summary(model)
# Realizamos predicciones
probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Sacamos la precisión del modelo
mean(predicted.classes == test.data$diabetes)


#Realizamos la regresión logística del modelo
model <- glm( diabetes ~ glucose, data = train.data, family = binomial)
summary(model)$coef

newdata <- data.frame(glucose = c(20,  180))
probabilities <- model %>% predict(newdata, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
predicted.classes

#Mostramos la función sigmoide del modelo
train.data %>%
  mutate(prob = ifelse(diabetes == "pos", 1, 0)) %>%
  ggplot(aes(glucose, prob)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Modelo de Regresión Logística", 
    x = "Concentración de glucosa en plasma",
    y = "Probabilidad de ser diabético positivo"
  )


#Realizamos una predicción
probabilities <- model %>% predict(test.data, type = "response")
head(probabilities)

predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
head(predicted.classes)

mean(predicted.classes == test.data$diabetes)

