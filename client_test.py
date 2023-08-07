import requests
import json
import cv2
import sys

api_url = "http://127.0.0.1:8000/face_detection/detect/"

image_path = "tes.jpg"
payload = {"url": image_path}
image = cv2.imread(image_path)

response = requests.post(api_url, data=payload, files={
    "image": open(image_path, "rb")})

if response.status_code == 200:
    data = response.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2))
    for face in data['faces']:
        curr_age = face['age']
        curr_gen = face['gender']
        curr_emo = face['emotion']
        curr_name = face['name']
        print()
        for (top, right, bottom, left) in face['location']:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(image, (left, bottom),
                          (right, bottom + 80), (0, 255, 0), -1)
            cv2.rectangle(image, (left, top),
                          (right, top - 50), (0, 255, 0), -1)

            cv2.putText(image, curr_name, (left + 20, top - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, curr_age, (left + 250, bottom + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, curr_gen, (left + 20, bottom + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, curr_emo, (left + 20, bottom + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.namedWindow("predicted image", cv2.WINDOW_NORMAL)
        cv2.imshow("predicted image", image)
        cv2.waitKey(0)
else:
    print("Request failed with status code:", response.status_code)
