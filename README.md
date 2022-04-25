# Image-Segmentation-using-KNN
Digital Image Processing

## Image Segmentation
Image segmentation is a method in which a digital image is broken down into various subgroups called Image segments which helps in reducing the complexity of the image to make further processing or analysis of the image simpler. Segmentation in easy words is assigning labels to pixels. All picture elements or pixels belonging to the same category have a common label assigned to them. For example: Let’s take a problem where the picture has to be provided as input for object detection. Rather than processing the whole image, the detector can be inputted with a region selected by a segmentation algorithm. This will prevent the detector from processing the whole image thereby reducing inference time.

# Approaches in Image Segmentation
Similarity approach: This approach is based on detecting similarity between image pixels to form a segment, based on a threshold. ML algorithms like clustering are based on this type of approach to segment an image.

Discontinuity approach: This approach relies on the discontinuity of pixel intensity values of the image. Line, Point, and Edge Detection techniques use this type of approach for obtaining intermediate segmentation results which can be later processed to obtain the final segmented image.

## KNN

K-Nearest Neighbor(KNN) Algorithm 
K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.
It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.
KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

## Our Proposed Work - 

In this poster, we proposed to implement a common and widely used image process- ing technique of Image Segementation using KNN Algorithm. As discussed 

above, here we initially define a K value which is the min no. of neighbours to be considered. The no. of K values must be odd to avoid ties. The K value 

for this poster is chosen to be 40, hence 40 neighbours will be considered before assigning the new pixel a cluster. A step by step walk around of the 

implemtation is:

The Number Program initially asks for the number of clusters to be selected in the K-NN algorithm. Obviously more the clusters more the number of 

segmented parts in the image.


So the steps Involved in K-NN algorithm:

Choosing the number of Clusters

Selecting at random K points for centroid, in our case 40 was passed as the number of neighbors.

Assigning each Data point as we say each pixel value closest to the above centroid that further gives us clusters.

Now we compute and place the new centroid for each cluster.

On the last step we just do the reassignment of the new nearest centroid and if in any case any new reassignment took place we would reiterate the above 
process.


## Pseudo Code - 

So the steps Involved in K-NN algorithm:

1. Import necessary libraries (eg. numpy, scikit, matplotlib)

2. Read and reshape the image

3. Choosing the number of Clusters

4. Selecting at random K points for centroid, in our case 40 was passed as the number of neighbors.

5. Assigning each Data point as we say each pixel value closest to the above centroid that further gives us clusters.

6. Now we compute and place the new centroid for each cluster.

7. On the last step we just do the reassignment of the new nearest centroid and if in

any case any new reassignment took place we would reiterate the above process.

All this procedure is condensed under one function in Sci-kit and hence making it easy to use

## Flowchart -

![Flowchart](https://user-images.githubusercontent.com/59500283/165037996-99e65cce-7edb-4d9e-b102-a9df2d50b983.jpg)


## Code - 

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

## Outputs -

![Img_1](https://user-images.githubusercontent.com/59500283/165038063-080c6a79-032b-4770-936d-8e386a89ce7e.jpeg)

![Img_2](https://user-images.githubusercontent.com/59500283/165038104-eb7715ed-19cb-4fbd-b6fd-5e129a37db61.jpeg)

![Img_3](https://user-images.githubusercontent.com/59500283/165038114-10e170f4-eb1d-4a07-b889-944e5a9e4e78.jpeg)

![Img_4](https://user-images.githubusercontent.com/59500283/165038129-8fb3c91a-0a6c-45df-9ffe-2e453c7a5200.jpeg)


## Results - 

This paper implements image segmentation for more comprehensive grouping and understanding of images and getting a more clearly distinct ROI. We applied image segmentation by the means of KNN on 2 images, one being a casual image and the other being an MRI scan. in both images we can see a significant form of clustering of regions of image which resume characteristics. For the normal image, the num- ber of clusters is selected to be 10 and for the MRI scan the number of sclusters is
