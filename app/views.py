from flask import render_template, send_file
from app import app
from app import kmeans
import cv2
import urllib
import numpy as np
from PIL import Image
from io import BytesIO

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'World'}	#fake user
    return render_template('index.html', title='Home', user=user)

@app.route('/kmeansapp')
def kmeansapp():
    return render_template('kmeans.html', title='K Means')

@app.route('/detect')
def detect():
    # Get image from a predefined URL
    image = _grab_image(url="https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/baboon_small.jpg")

    # Fix the number of clusters
    k = 3

    # Do k means clustering
    result = kmeans.kmeansclustering(image, k)

    # Display the resulting image
    img = Image.fromarray(result)

    byte_io = BytesIO()
    img.save(byte_io, 'PNG')
    byte_io.seek(0)

    return send_file(byte_io, mimetype='image/png')

def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image