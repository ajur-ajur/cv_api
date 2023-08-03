import cv2
import requests
from face_detector import views as v


url = 'http://localhost:8000/face_detection/detect/'

image = cv2.imread("images/obama.jpg")
payload = {"image": open("images/obama.jpg", "rb")}

r = requests.post(url, files=payload).json()

print("obama.jpg: {}".format(r))

for (top, right, bottom, left) in r["faces"]:
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

images = cv2.resize(image, (600, 600))
cv2.imshow("images/obama.jpg", images)
cv2.waitKey(0)
