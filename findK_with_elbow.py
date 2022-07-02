import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Load Img
img = plt.imread("ex_img.jpg")

#Reshape Image
height = img.shape[0]
width = img.shape[1]
img = img.reshape(width*height, 4)

#Create Elbow
errors = []
k_values = []
for i in range(1,50):
	k_values.append(i)
	kmeans = KMeans(n_clusters=i).fit(img)
	kmeans.predict(img)
	errors.append(kmeans.inertia_)
plt.plot(k_values, errors, marker="*")
plt.xlabel("K Values")
plt.ylabel("Errors")
plt.title("Find K with Elbow")
plt.show()