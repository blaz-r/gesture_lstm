import tensorflow.python.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import mediapipe as mp
import cv2

import os
from pathlib import Path
import re

import depthai as dai
from dai_utils import create_pipeline, find_isp_scale_params


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

resolution = (1920, 1080)

# scale image, this better in my case than native 1920 x 1080
internal_frame_height = 640
width, scale_nd = find_isp_scale_params(
    internal_frame_height * resolution[0] / resolution[1],
    resolution,
    is_height=False)

img_h = int(round(resolution[1] * scale_nd[0] / scale_nd[1]))
img_w = int(round(resolution[0] * scale_nd[0] / scale_nd[1]))
pad_h = (img_w - img_h) // 2
frame_size = img_w


def extract_landmarks(results):
    """
    Extract landmarks from mediapipe result

    :param results: mediapipe hands result file
    :return: landmark array, 42 in image coordiantes
    """
    landmarks = np.zeros(42)

    lms = results.multi_hand_landmarks[0]
    for idx, landmark in enumerate(lms.landmark):
        landmarks[idx * 2] = landmark.x * img_w
        landmarks[idx * 2 + 1] = landmark.y * img_h

    landmarks /= 5

    return landmarks


def produce_landmarks(gesture_videos_path, gesture_landmarks_path, name):
    """
    Process saved videos to produce landmark data

    :param gesture_videos_path: path to gesture video directory
    :param gesture_landmarks_path: path where gesture ladnmarks will be saved
    :param name: name of files that will contain landmarks
    :return: None
    """
    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands:

        for file_name in os.listdir(gesture_videos_path):
            file_path = f"{gesture_videos_path}/{file_name}"
            # open video file
            cap = cv2.VideoCapture(file_path)

            # 30 frames with 42 landmarks
            landmarks = np.zeros((30, 42))

            frame_idx = 0
            successful = False
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    frame.flags.writeable = False
                    results = hands.process(frame)
                    frame.flags.writeable = True

                    if not results.multi_hand_landmarks:
                        print(f"{file_path} has no detection")
                        break

                    # insert scaled landmarks for current frame
                    landmarks[frame_idx, :] = extract_landmarks(results)
                    frame_idx += 1
                else:
                    # end of video
                    successful = True
                    break

            if successful:
                no_ext_file_name = Path(file_name).stem
                num = re.findall(r'\d+', no_ext_file_name)[0]
                # if while didn't exit due to no landmark detections
                np_path = f"{gesture_landmarks_path}/{name}_{num}"
                np.save(np_path, landmarks)

            cap.release()


def landmarks_read_test(directory_path):
    """
    Test reading landmarks

    :param directory_path: path to directory containing landmark data in .npy format
    :return:
    """
    for idx, file in enumerate(os.listdir(directory_path)):
        file_path = f"{directory_path}/{file}"

        res = np.load(file_path)

        print("Successfully read")


def prepare_data(path):
    """
    Prepare landmark data for training

    :param path: path to directory with subdirectories containing landmarks
    :return: sequences (X), results (y)
    """
    gestures = ["play", "pause", "idle"]
    gestures_onehot_dict = {"play": [1, 0, 0],
                            "pause": [0, 1, 0],
                            "idle": [0, 0, 1]}

    sequences = []
    results = []
    for gesture in gestures:
        gesture_path = f"{path}/{gesture}"
        result = gestures_onehot_dict[gesture]
        for file in os.listdir(gesture_path):
            file_path = f"{gesture_path}/{file}"
            landmarks = np.load(file_path)

            sequences.append(landmarks)
            results.append(result)

    sequences = np.array(sequences)
    results = np.array(results)

    return sequences, results


def make_model():
    """
    Build model architecture

    :return:
    """
    model = Sequential()
    model.add(Input(shape=(30, 42), name="input"))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax', name="result"))

    return model


def train_lstm():
    """
    Train model from saved landmarks

    :return:
    """
    X, y = prepare_data("gesture_landmarks")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = make_model()

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=600)

    model.summary()

    model.save("gestures.h5")

    y_hat = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_hat = np.argmax(y_hat, axis=1).tolist()

    print("AC on test data",accuracy_score(y_true, y_hat))

    return model


def test_model():
    """
    Test model weight loading

    :return:
    """
    X, y = prepare_data("gesture_landmarks")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = make_model()
    model.load_weights("gestures.h5")

    y_hat = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_hat = np.argmax(y_hat, axis=1).tolist()

    print(accuracy_score(y_true, y_hat))


def live_test():
    """
    Test gesture recognition live

    :return:
    """
    # load model architecture and weight
    model = make_model()
    model.load_weights("gestures.h5")

    gestures = ["play", "pause", "idle"]

    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands:

        sequence = []
        predictions = []
        threshold = 0.5
        sentence = []

        while True:

            image = img_q.get().getCvFrame()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                # extract and sacale landmarks
                landmarks = extract_landmarks(results)
                sequence.append(landmarks)
                # keep last 30 frames
                sequence = sequence[-30:]

            if len(sequence) == 30:
                # predict with lstm
                result = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(result))
                print(result)

                # output if last 20 frames are all same prediction
                if np.unique(predictions[-20:])[0] == np.argmax(result):
                    if result[np.argmax(result)] > threshold:

                        if len(sentence) > 0:
                            if gestures[np.argmax(result)] != sentence[-1]:
                                sentence.append(gestures[np.argmax(result)])
                        else:
                            sentence.append(gestures[np.argmax(result)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            key = cv2.waitKey(5)
            if key == 27 or key == ord('q'):
                break


def live_mediapipe():
    """
    Mediapipe hand detection from live camera

    :return: None
    """
    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands:
        while True:

            image = img_q.get().getCvFrame()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('MediaPipe Hands', image)
            key = cv2.waitKey(5)
            if key == 27 or key == ord('q'):
                break


def main():
    name = "idle"
    lm_path = f"gesture_landmarks/{name}"
    video_path = f"gesture_videos/{name}"

    produce_landmarks(video_path, lm_path, name)
    # landmarks_read_test(lm_path)
    # prepare_data("gesture_landmarks")


if __name__ == '__main__':
    # main()
    # live_mediapipe()
    # train_lstm()
    # test_model()
    live_test()
