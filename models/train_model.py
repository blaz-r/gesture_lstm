import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import numpy as np
import mediapipe as mp
import cv2

import os
from pathlib import Path
import re
import time

from dai_utils import find_isp_scale_params

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

    landmarks /= frame_size

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


def prepare_data(path, gestures, gestures_onehot_dict, train_labels):
    """
    Prepare landmark data for training

    :param path: path to directory with subdirectories containing landmarks
    :param gestures: list of gestures, in order which model predicts them
    :param gestures_onehot_dict: dictionary mapping gesture to prediction output
    :param train_labels: array containing id of people performing gestures to be included in trianing
    :return: sequences (X), results (y)
    """
    sequences = []
    results = []
    for gesture in gestures:
        gesture_path = f"{path}/{gesture}"
        result = gestures_onehot_dict[gesture]
        for file in os.listdir(gesture_path):
            # skip files that are from person that shouldn't be included in training
            if file.split("_")[1] not in train_labels:
                continue

            file_path = f"{gesture_path}/{file}"
            landmarks = np.load(file_path)

            sequences.append(landmarks)
            results.append(result)

    sequences = np.array(sequences)
    results = np.array(results)

    return sequences, results


def make_model(outputs=6):
    """
    Build model architecture

    :return:
    """
    model = Sequential()
    model.add(Input(shape=(30, 42), name="input"))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(outputs, activation='softmax', name="result"))

    return model


def train_lstm(gestures, gestures_onehot_dict, train_labels=["0", "1"]):
    """
    Train model from saved landmarks

    :param gestures: list of gestures, in order which model predicts them
    :param gestures_onehot_dict: dictionary mapping gesture to prediction output
    :param train_labels: array containing id of persons performing gestures to be included in trianing
    :return:
    """

    # train on CPU since it's faster for this case
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    X, y = prepare_data("gesture_landmarks", gestures, gestures_onehot_dict, train_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = make_model(len(y[0]))

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=300)

    model.summary()

    y_hat = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_hat = np.argmax(y_hat, axis=1).tolist()

    print("AC on validation data", accuracy_score(y_true, y_hat))

    MODEL_DIR = "gesture_recognition_lstm"
    tf.saved_model.save(model, MODEL_DIR)

    model.save("gestures.h5")

    return model


def test_model(model_path, path, gestures, gestures_onehot_dict):
    """
    :param path: path to directory with subdirectories containing landmarks
    :param gestures: list of gestures, in order which model predicts them
    :param gestures_onehot_dict: dictionary mapping gesture to prediction output

    :return:
    """
    X_test, y_test = prepare_data(path, gestures, gestures_onehot_dict, ["1"])

    model = make_model(6)
    model.load_weights(model_path)

    y_hat = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_hat = np.argmax(y_hat, axis=1).tolist()

    print(accuracy_score(y_true, y_hat))

    y_true_labels = [gestures[i] for i in y_true]
    y_hat_labels = [gestures[i] for i in y_hat]

    cm = confusion_matrix(y_true_labels, y_hat_labels, labels=gestures)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gestures)
    disp.plot()
    plt.show()


def run_test_data(path, gestures):
    # train on CPU since it's faster for this case
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = make_model(6)
    model.load_weights("gestures_2people_95.h5")

    y_true = []
    y_hat = []

    total_time_ns = 0

    for g_index, gesture in enumerate(gestures):
        gesture_path = f"{path}/{gesture}"
        for file in os.listdir(gesture_path):
            file_path = f"{gesture_path}/{file}"
            landmarks = np.load(file_path)

            y_true.append(g_index)

            data = np.expand_dims(landmarks, axis=0)

            t0 = time.time_ns()
            result = model.predict(data, verbose=0)
            t1 = time.time_ns()

            total_time_ns += t1 - t0

            y_hat.append(np.argmax(result[0]))

    print("AC on test data", accuracy_score(y_true, y_hat))

    return total_time_ns / (10 ** 9)


def test_on_landmarks():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\pause\pause_0_0.npy")
    data = np.expand_dims(data, axis=0)

    model = make_model(6)
    model.load_weights("gestures_2people_95.h5")

    t0 = time.time_ns()
    res = model.predict(data)
    t1 = time.time_ns()

    print(t1 - t0, res)


def save_model():
    model = make_model()
    model.load_weights("gestures.h5")

    MODEL_DIR = "gesture_recognition_lstm"
    tf.saved_model.save(model, MODEL_DIR)


def check_weights():
    model = make_model()
    model.load_weights("gestures.h5")

    weights = model.get_weights()
    print(weights)


def process_landmarks():
    name = "forward"
    lm_path = f"gesture_landmarks/{name}"
    video_path = f"gesture_videos/{name}"

    # produce_landmarks(video_path, lm_path, name)
    # landmarks_read_test(lm_path)
    # prepare_data("gesture_landmarks")


def avg_time(gestures):
    time = 0

    for i in range(10):
        time += run_test_data("../test_data/test/gesture_landmarks", gestures)

    print("Average execution time in seconds: ", time / 10)


def main():
    gestures = ["play", "pause", "forward", "back", "idle", "vol"]

    gestures_onehot_dict = {name: [1 if gestures.index(name) == i else 0 for i in range(len(gestures))]
                            for name in gestures}

    # gestures_onehot_dict = {"play": [1, 0, 0, 0, 0],
    #                         "pause": [0, 1, 0, 0, 0],
    #                         "forward": [0, 0, 1, 0, 0],
    #                         "back": [0, 0, 0, 1, 0],
    #                         "idle": [0, 0, 0, 0, 1]}

    # train_lstm(gestures, gestures_onehot_dict, train_labels=["0"])
    # save_model()
    # check_weights()
    # test_model("gestures.h5", "test_data/test/gesture_landmarks", gestures, gestures_onehot_dict)
    avg_time(gestures)

    # test_on_landmarks()


if __name__ == '__main__':
    main()
    # process_landmarks()
