import numpy as np
import os
import face_recognition as fr
from keras.models import load_model
from django.db import models


model_age = load_model('face_detector/model/agegender.h5')
model_emotion = load_model('face_detector/model/emotion.h5')
model_name = load_model('face_detector/model/fine_tuning.h5')


def get_age(distr):
    if distr >= 0.01 and distr <= 0.10:
        return "9-18"
    if distr >= 0.11 and distr <= 0.30:
        return "19-25"
    if distr >= 0.31 and distr <= 0.35:
        return "26-37"
    if distr >= 0.36 and distr <= 0.40:
        return "38-49"
    if distr >= 0.60:
        return "60 +"
    return 'Unknown'


def get_gender(prob):
    if prob > 0.5:
        return "Male"
    else:
        return "Female"


def get_emotion(hrr):
    emotion_dict = {0: "Angry", 1: "Happy", 2: "Disgust",
                    3: "Surprise", 4: "Sad", 5: "Fear", 6: "Neutral"
                    }

    maxindex = int(np.argmax(hrr))
    return emotion_dict[maxindex]


def get_name(pred):
    name_dict = {0: 'aldira', 1: 'ridho', 2: 'sabila'}

    maxindex = int(np.argmax(pred))
    return name_dict[maxindex]


def face_locations(frame):
    face_locations = []
    face_location = fr.face_locations(frame)
    face_locations.append(face_location)

    return face_locations


class FaceData(models.Model):
    top = models.IntegerField()
    right = models.IntegerField()
    bottom = models.IntegerField()
    left = models.IntegerField()
    emotion = models.CharField(max_length=50)
    age = models.CharField(max_length=20)
    gender = models.CharField(max_length=10)
