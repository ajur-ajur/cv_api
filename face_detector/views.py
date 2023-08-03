from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import cv2
import face_recognition as fr
from face_detector import grab as g
from face_detector import models as m


@csrf_exempt
def detect(request):
    data = {"success": False}
    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image = g._grab_image(stream=request.FILES["image"])
        else:
            url = request.POST.get("url", None)
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            image = g._grab_image(url=url)

        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        face_locations = fr.face_locations(small_image)

        for (top, right, bottom, left) in face_locations:
            ROI = image[top+20:bottom+20, left+20:right+20]

        image_grey = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        input_image = np.expand_dims(np.expand_dims(
            cv2.resize(image_grey, (48, 48)), -1), 0)

        predict_emotion = m.model_emotion.predict(input_image)
        predict_age = m.model_age.predict(input_image)

        print(predict_age[0])

        data.update({"faces": face_locations, "success": True,
                     "emotion": m.get_emotion(predict_emotion),
                     "age": m.get_age(predict_age[0]),
                     "gender": m.get_gender(predict_age[1]),
                     "top": top,
                     "right": right,
                     "left": left,
                     "bottom": bottom,
                     })

    return JsonResponse(data)
