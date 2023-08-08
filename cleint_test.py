import requests
import json
import cv2

api_url = "http://127.0.0.1:8000/face_detection/detect/"
count = 0
des = 90  # 30 = 1 detik

vid = cv2.VideoCapture(0)

while (True):
    ret, frame = vid.read()
    count += 1

    if (des / count == 1):
        image_path = "frame.jpg"

        cv2.imwrite(image_path, frame)
        payload = {"url": image_path}

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
                # cv2.namedWindow("predicted image", cv2.WINDOW_NORMAL)
                # cv2.imshow("predicted image", frame)
                # cv2.waitKey(0)

            count = 0

        else:
            print("Request failed with status code:", response.status_code)

    for (top, right, bottom, left) in face['location']:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top),
                      (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom),
                      (right, bottom + 80), (0, 255, 0), -1)
        cv2.rectangle(frame, (left, top),
                      (right, top - 50), (0, 255, 0), -1)

        cv2.putText(frame, curr_name, (left + 20, top - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, curr_age, (left + 250, bottom + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, curr_gen, (left + 20, bottom + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, curr_emo, (left + 20, bottom + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
