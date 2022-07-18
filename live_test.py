import os

import depthai as dai
import onnxruntime
from openvino.inference_engine import IECore
from dai_utils import create_pipeline, find_isp_scale_params
import numpy as np
import mediapipe as mp
import cv2

import pyautogui

from train_model import extract_landmarks, make_model
import time

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


class Model:
    def __init__(self):
        pass

    def predict(self, sequence):
        pass


class TfModel(Model):
    def __init__(self):
        # load model architecture and weight
        super().__init__()
        self.model = make_model()
        self.model.load_weights("gestures.h5")

    def predict(self, sequence):
        return self.model.predict(sequence)[0]


class OnnxModel(Model):
    def __init__(self):
        super().__init__()
        self.model = "gesture_recognition_lstm.onnx"

        self.session = onnxruntime.InferenceSession(self.model, None)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, sequence):
        return self.session.run([self.output_name], {self.input_name: sequence})[0][0]


class OpenVinoModel(Model):
    def __init__(self):
        super().__init__()
        self.iecore = IECore()

        # load model architecture and weight
        self.network = self.iecore.read_network(model="gesture_recognition_lstm.xml", weights="gesture_recognition_lstm.bin")
        self.exec_net = self.iecore.load_network(network=self.network, device_name="CPU")
        self.input_blob = next(iter(self.network.input_info))

    def predict(self, sequence):
        return self.exec_net.infer(inputs={self.input_blob: np.expand_dims(sequence, axis=0)})["result"][0]


def gesture_to_command(gesture, prev_gesture, landmarks):
    """
    Convert gesture string to actual command string

    :param gesture: gesture string
    :param prev_gesture: previous gesture
    :param landmarks: sequence of landmarks
    :return: command string or None if idle or same gesture
    """
    if gesture == prev_gesture:
        return None

    if gesture == "play":
        return "playpause"
    elif gesture == "pause":
        return "playpause"
    elif gesture == "back":
        return "prevtrack"
    elif gesture == "forward":
        return "nexttrack"
    elif gesture == "vol":
        # * 2 cuz we have 2 coords and + 1 as we want y coord, index finger is on index 8
        index_finger_pos = landmarks[8 * 2 + 1]

        if index_finger_pos < (648 / 1152) * 0.5:
            return "volumeup"
        else:
            return "volumedown"
    else:
        return None


def live_test(gestures, model, unique_limit=15, threshold=0.5):
    """
    Test gesture recognition live

    :param gestures: list of gestures
    :param model: model used for recognition
    :param unique_limit: number of detections that need to be all same for it to count as a gesture
    :param threshold: threshold for probability that gesture was really detected
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

        sequence = []
        predictions = []
        pred_probs = []
        sentence = []

        prev_gesture = ""

        while True:

            image = img_q.get().getCvFrame()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            hand_result = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True

            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                # extract and sacale landmarks
                landmarks = extract_landmarks(hand_result).astype(np.float32)
                sequence.append(landmarks)
                # keep last 30 frames
                sequence = sequence[-30:]

            if len(sequence) == 30:
                # predict with lstm
                t0 = time.time()
                result = model.predict(np.expand_dims(sequence, axis=0))
                t1 = time.time()

                pred_index = np.argmax(result)

                predictions.append(pred_index)
                pred_probs.append(result[pred_index])

                print(round(t1 - t0, 4), [round(prob, 3) for prob in result])

                # output if last 10 frames are all same prediction
                unique = np.unique(predictions[-unique_limit:])

                current_gesture = "idle"

                if len(unique) == 1 and unique[0] == pred_index:
                    if np.all(prob > threshold for prob in pred_probs[-unique_limit:]):
                        current_gesture = gestures[pred_index]

                if len(sentence) > 0:
                    if current_gesture != sentence[-1]:
                        sentence.append(current_gesture)
                else:
                    sentence.append(current_gesture)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                command = gesture_to_command(current_gesture, prev_gesture, landmarks)
                if command:
                    pyautogui.press(command)

                prev_gesture = current_gesture

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
            min_detection_confidence=0.1,
            min_tracking_confidence=0.3) as hands:
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


def long_test(gestures, model, unique_limit=15, threshold=0.5):
    """
    Test system on saved long test sequence

    :param gestures: list of gestures
    :param model: model used for recognition
    :param unique_limit: number of detections that need to be all same for it to count as a gesture
    :param threshold: threshold for probability that gesture was really detected
    :return: None
    """
    path = "test_data/long_test/landmarks"

    sequence = []
    predictions = []
    pred_probs = []
    sentence = []

    prev_gesture = ""

    for idx, file in enumerate(os.listdir(path)):
        file_path = f"{path}/{file}"
        landmarks = np.load(file_path)
        landmarks = landmarks.astype(np.float32)

        sequence.append(landmarks)
        # keep last 30 frames
        sequence = sequence[-30:]

        if len(sequence) == 30:
            # predict with lstm
            result = model.predict(np.expand_dims(sequence, axis=0))

            pred_index = np.argmax(result)

            predictions.append(pred_index)
            pred_probs.append(result[pred_index])

            # output if last 10 frames are all same prediction
            unique = np.unique(predictions[-unique_limit:])

            current_gesture = "idle"

            if len(unique) == 1 and unique[0] == pred_index:
                if np.all(prob > threshold for prob in pred_probs[-unique_limit:]):
                    current_gesture = gestures[pred_index]

            if len(sentence) > 0:
                if current_gesture != sentence[-1]:
                    sentence.append(current_gesture)
            else:
                sentence.append(current_gesture)

            prev_gesture = current_gesture

    print(sentence)


if __name__ == '__main__':
    # gestures = ["play", "pause", "forward", "back", "idle"]
    gestures = ["play", "pause", "forward", "back", "idle", "vol"]
    # live_mediapipe()
    # live_test(gestures, TfModel())
    # live_test(gestures, OpenVinoModel())
    # live_test(gestures, OnnxModel(), unique_limit=20, threshold=0.5)
    long_test(gestures, OnnxModel(), unique_limit=20, threshold=0.5)
