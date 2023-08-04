import requests
import json
import cv2

api_url = "http://127.0.0.1:8100/face_detection/detect/"  # Update with your API URL

image_path = "images/obama.jpg"
payload = {"url": image_path}
image = cv2.imread(image_path)

response = requests.post(api_url, data=payload, files={
    "image": open(image_path, "rb")})

if response.status_code == 200:
    data = response.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2))
else:
    print("Request failed with status code:", response.status_code)


'''
import cv2
import requests


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

'''
