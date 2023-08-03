import cv2
import sys
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import pickle

concat = 'aldira'
img_dir = 'images/'+concat+'.jpg'
cap = cv2.VideoCapture(1)
frame = cap.read()[1]
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# def encode2str(frame):
#     '''Encoder'''
#     cnt = cv2.imencode('.png', frame)[1]
#     b64 = base64.b64encode(cnt)

#     return b64


# def decode2img(b64):
#     ''''Decoder'''
#     frame = base64.b64decode(b64)
#     nparr = np.frombuffer(frame, dtype=np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     return img


# str = encode2str(frame)
# img = decode2img(str)

# image = Image.fromarray(img, 'RGB')
# image.save('test.png')
# image.show

# image = cv2.imread(frame)

# print(image)
# print(frame)


# def im2json(im):
#     _, imdata = cv2.imencode('.JPG', im)
#     jstr = json.dumps({"image": base64.b64encode(imdata).decode('ascii')})
#     return jstr


# def json2im(jstr):
#     load = json.loads(jstr)
#     imdata = base64.b64decode(load['image'])
#     im = Image.open(BytesIO(imdata))
#     return im


# red_img = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)

# jstr = im2json(frame)

# PILimage = json2im(jstr)
# PILimage.show()

def im2json(im):
    '''Funtion to encode image to JSON string'''
    imdata = pickle.dumps(im)
    jstr = json.dumps({'image': base64.b64encode(imdata).decode('ascii')})
    return jstr


def json2im(jstr):
    '''Function to decode JSON string to image'''
    load = json.loads(jstr)
    imdata = base64.b64decode(load['image'])
    im = pickle.loads(imdata)
    return im


encoded = im2json(frame)
decoded = json2im(encoded)

image = Image.fromarray(decoded, 'RGB')
image.save('test.png')

print(encoded)
print(decoded)
