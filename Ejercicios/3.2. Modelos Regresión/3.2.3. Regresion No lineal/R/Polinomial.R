#Ejemplo del modelo de Regresión No Lineal o Polinomial

#Creamos unos valores aleatorios que se adecuen a nuestro modelo.
peq <- function(x) x^3+2*x^2+5

x <- seq(-0.99, 1, by = .01)
y <- peq(x) + runif(200)

df <- data.frame(x = x, y = y)
head(df)

#Visualizamos únicamente los datos en nuestro gráfico
plot(df$x, df$y, pch=20, col="red",xlab = "Variable independiente", ylab = "Variable dependiente")

#Creamos el modelo y predecimos
model <- lm(y~x+I(x^3)+I(x^2), data = df)
summary(model)

pred <- predict(model,data=df)

#Visualizamos los resultados del modelo.  

windows(width=8, height=6)
plot(x=df$x, y=df$y, pch=20, col="grey",xlab = "Variable independiente", ylab = "Variable dependiente")

lines(df$x, predict(lm(y~x, data=df)), type="l", col="orange", lwd=2)
lines(df$x, predict(lm(y~I(x^2), data=df)), type="l", col="pink", lwd=2)
lines(df$x, predict(lm(y~I(x^3), data=df)), type="l", col="yellow", lwd=2)
lines(df$x, predict(lm(y~poly(x,3)+poly(x,2), data=df)), type="l", col="blue", lwd=3)

legend("topleft", 
       legend = c("Lineal","Cuadrática", "Cúbica", "Polinomial"), 
       col = c("orange","pink","yellow","blue"),
       lty = 1, lwd=3
) 
