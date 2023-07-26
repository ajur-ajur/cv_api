# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from face_detector import grab as t
from face_detector import models as m
import face_recognition as fr
import cv2
import numpy as np
import os

FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_alt2.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	data = {"success": False}
	if request.method == "POST":
		if request.FILES.get('image', None) is not None:
			image = t._grab_image(stream=request.FILES['image'])
		else:
			url = request.POST.get("url", None)
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
			image = t._grab_image(url=url)
		### START WRAPPING OF COMPUTER VISION APP
		
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #pake	
		small_img = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)

		m.encode_faces()

		face_locations = fr.face_locations(small_img)
		face_encodings = fr.face_encodings(small_img, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			matches = fr.compare_faces(m.known_face_encodings, face_encoding)
			name = "Unknown"
			confidence = '???'

			face_distances = fr.face_distance(m.known_face_encodings, face_encoding)

			best_match_index = np.argmax(face_distances)
			if matches[best_match_index]:
				name = m.known_face_names[best_match_index]
				confidence = m.face_confidence(
					face_distances[best_match_index])

			face_names.append(f'{name} ({confidence})')


		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
		rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

		# zipped = zip(face_locations, face_names)
		# rects = [(int(x), int(y), int(w), int(h)) for (x, y, w, h), name in zipped]

		emotion_model = m.emotion_model
		age_model = m.age_model

		emo_image = np.expand_dims(np.expand_dims(cv2.resize(image_gray, (48, 48)), -1), 0)
		emo_prediction = emotion_model.predict(emo_image)
		age_prediction = age_model.predict(emo_image)
		maxindex = int(np.argmax(emo_prediction))


		data.update({"success": True, "num_faces": len(rects), "faces": rects,
		"emotion": m.emotion_dict[maxindex], "gender": m.get_gender(age_prediction[1]), "age": m.get_age(age_prediction[0]),
		"name": name})

		### END WRAPPING OF COMPUTER VISION APP
		data["success"] = True
	return JsonResponse(data)
