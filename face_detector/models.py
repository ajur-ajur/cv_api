import face_recognition as fr
import math
import os
from keras.models import load_model

age_dir = 'face_detector/model/agegender.h5'
emotion_dir = 'face_detector/model/emotion.h5'
emotion_model = load_model(emotion_dir)
age_model = load_model(age_dir)
emotion_dict = {0: "Angry", 1: "Happy", 2: "Disgust",
                3: "Surprise", 4: "Sad", 5: "Fear", 6: "Neutral"}


def face_confidence(face_distance, face_match_threshold=0.1):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


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
    return "Unknown"


def get_gender(prob):
    if prob < 0.5:
        return "Male"
    else:
        return "Female"


def multiply_list_of_tuples(list_of_tuples, factor):
    result = []
    for tpl in list_of_tuples:
        multiplied_tuple = tuple(item * factor for item in tpl)
        result.append(multiplied_tuple)
    return result
