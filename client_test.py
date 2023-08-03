import requests
import cv2
import base64
import json
import pickle

url = 'http://localhost:8000/face_detection/detect/'

font = cv2.FONT_HERSHEY_DUPLEX

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def im2json(im):
    '''Funtion to encode image to JSON string'''
    imdata = pickle.dumps(im)
    jstr = json.dumps({'image': base64.b64encode(imdata).decode('ascii')})
    return jstr


load = im2json(frame)

payload = {'str': load}

r = requests.post(url, json=payload).json()

print(f"Status Code: {r.status_code}, Response: {r.json()}")
print("Prediction: {}".format(r))

for (startX, startY, endX, endY) in r['faces']:
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # masih ada bug dalam deteksi nama, mungkin harus ganti model
    cv2.putText(frame, "Nama: {}".format(
        r['name']), (startX + 20, startY + - 15), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Gender: {}".format(
        r['gender']), (startX + 20, startY + 30), font, 1, (0, 255, 0), 1)
    cv2.putText(frame, "Usia: {}".format(
        r['age']), (startX + 20, startY + 60), font, 1, (0, 255, 0), 1)
    cv2.putText(frame, "Emosi: {}".format(
        r['emotion']), (startX + 20, startY + 90), font, 1, (0, 255, 0), 1)

cv2.namedWindow("prediksi", cv2.WINDOW_NORMAL)
cv2.imshow("prediksi", frame)
cv2.waitKey(0)
