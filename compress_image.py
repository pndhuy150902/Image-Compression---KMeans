import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Load Image
img = plt.imread("ex_img.jpg")

#Reshape Image
height = img.shape[0]
width = img.shape[1]
img = img.reshape(width*height, 4)

#Use KMeans
kmeans = KMeans(n_clusters=5).fit(img)
#Use 18 colors for Image
# kmeans = KMeans(n_clusters=18).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

#Create new Image
new_img = np.zeros_like(img)
for i in range(len(new_img)):
	new_img[i] = clusters[labels[i]]
new_img = new_img.reshape(height, width, 4)
plt.imshow(new_img)
plt.show()