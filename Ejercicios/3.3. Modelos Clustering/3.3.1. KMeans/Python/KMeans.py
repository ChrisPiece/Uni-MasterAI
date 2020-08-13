#Ejemplo de algoritmo de clustering para K - Means

#Importamos librerías y dataset.
import matplotlib.pyplot as plt
import pandas as pd

#Importamos el dataset y seleccionamos las columnas de Ingresos y el valor de puntuación del gasto (Income y Spending Score)
dataset= pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:, [3,4]].values


from sklearn.cluster import KMeans

#Calculamos el "punto de codo" para conocer el número de cluster ideal para nuestros datos
#Este método usa una función matemática conocida como Within Cluster Sum of Squares - WCSS (Suma de las distancias desde sus centroides)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Método "Punto de Codo"')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.show()
#Comprobamos que el número de cluster ideal está entre el 4 y el 6. Seleccionamos el valor 5 para K


kmeans = KMeans(n_clusters=5, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
#Si queremos probar con 7 cluster tendremos que cambiar n_clusters=7
#plt.scatter(X[y_kmeans==5, 0], X[y_kmeans==5, 1], s=100, c='orange', label ='Cluster 6')
#plt.scatter(X[y_kmeans==6, 0], X[y_kmeans==6, 1], s=100, c='brown', label ='Cluster 7')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters de Clientes')
plt.xlabel('Ingresos anuales')
plt.ylabel('Puntuación de Gasto(1-100')
plt.show()