import depthai as dai
import onnxruntime
from openvino.inference_engine import IECore
from dai_utils import create_pipeline, find_isp_scale_params
import numpy as np
import mediapipe as mp
import cv2

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


def live_test(gestures, model):
    """
    Test gesture recognition live

    :return:
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
                landmarks = extract_landmarks(results).astype(np.float32)
                sequence.append(landmarks)
                # keep last 30 frames
                sequence = sequence[-30:]

            if len(sequence) == 30:
                # predict with lstm
                t0 = time.time()
                result = model.predict(np.expand_dims(sequence, axis=0))
                t1 = time.time()
                predictions.append(np.argmax(result))

                print(t1 - t0, result)

                # output if last 10 frames are all same prediction
                unique = np.unique(predictions[-10:])
                if len(unique) == 1 and unique[0] == np.argmax(result):
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


if __name__ == '__main__':
    gestures = ["play", "pause", "forward", "back", "idle"]
    # live_mediapipe()
    # live_test(gestures, TfModel())
    # live_test(gestures, OpenVinoModel())
    live_test(gestures, OnnxModel())
