from sklearn.cluster import KMeans

import matplotlib as mpl
mpl.use('TkAgg')

import cv2

def kmeansclustering(image, k):
    # Convert from BGR to LAB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Convert to list of pixels for clustering
    pixellist = image.reshape((image.shape[0] * image.shape[1], 3))

    # Perform kmeans clustering
    cluster = KMeans(n_clusters=k)
    labels = cluster.fit_predict(pixellist)
    quant = cluster.cluster_centers_.astype("uint8")[labels]

    # Reshape feature vectors back to form the image
    quant = quant.reshape((image.shape[0], image.shape[1], 3))

    # Convert from LAB to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)

    return quant