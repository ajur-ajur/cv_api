import numpy as np
from keras.models import load_model
from django.db import models


model_age = load_model('face_detector/model/agegender.h5')
model_emotion = load_model('face_detector/model/emotion.h5')


def get_age(distr):
    if distr >= 1 and distr <= 10:
        return "9-18"
    if distr >= 11 and distr <= 30:
        return "19-25"
    if distr >= 31 and distr <= 35:
        return "26-37"
    if distr >= 36 and distr <= 40:
        return "38-49"
    if distr >= 60:
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


class FaceData(models.Model):
    top = models.IntegerField()
    right = models.IntegerField()
    bottom = models.IntegerField()
    left = models.IntegerField()
    emotion = models.CharField(max_length=50)
    age = models.CharField(max_length=20)
    gender = models.CharField(max_length=10)
