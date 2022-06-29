import tensorflow.python.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import numpy as np
import mediapipe as mp
import cv2

def make_landmarks():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        cap = cv2.VideoCapture("G:\Vids\Davinci\geste_test\CutTest.mp4")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        print(f"{frame_width} x {frame_height}")

        progress = 0
        while cap.isOpened():
            ret, frame = cap.read()

            progress += 1
            print(f"Progress: {progress}/{frame_count}")

            if ret:
                cv2.imshow("Read", frame)
                key = cv2.waitKey(20)
            else:
                break

        cap.release()
