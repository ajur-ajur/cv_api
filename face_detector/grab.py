# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import base64

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
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