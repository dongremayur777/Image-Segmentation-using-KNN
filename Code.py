import numpy as np 
from sklearn.cluster import KMeans
import datetime
import cv2
from matplotlib import pyplot as plt
import time

from sklearn.decomposition import PCA


img = cv2.imread(r"IIIT-Nagpur.jpg")
#img2 = img.reshape((-1,3))

reshaped = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

numClusters = int(input("ENTER THE NUMBER OF CLUSTERS:- "))

kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)
clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(img.shape[0], img.shape[1]))

sortedLabels = sorted([n for n in range(numClusters)],
        key=lambda x: -np.sum(clustering == x))

#kmeansImage = []

kmeansImage = np.zeros(img.shape[:2], dtype=np.uint8)

for i, label in enumerate(sortedLabels):
    kmeansImage[ clustering == label ] = int((255) / (numClusters - 1)) * i
cv2.imshow('window', kmeansImage)
cv2.imwrite("ACADEMICSEG.jpg", kmeansImage)


pca = PCA(2)

#Transform the data
df = pca.fit_transform(reshaped)
print(df.shape)

#Getting the Centroids
centroids = kmeans.cluster_centers_

 
#plotting the results:
plt.subplot(2,1,1) 
for i in sortedLabels:
    plt.scatter(df[label == i , 1] , df[label == i , 0] , label = i)
plt.scatter(centroids[:,1] , centroids[:,0] , s = 80, color = 'k')

plt.legend()

plt.subplot(2,1,2)
for i in sortedLabels:
    plt.scatter(df[label == i , 1] , df[label == i , 0] , color = 'blue')

plt.legend()

plt.show()

#plt.savefig('MRI1SCT.png')



#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows()