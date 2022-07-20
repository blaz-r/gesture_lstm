import numpy as np
from openvino.inference_engine import IECore
import time
from sklearn.metrics import accuracy_score
import os

iecore = IECore()

network = iecore.read_network(model="gesture_recognition_lstm_2p95.xml", weights="gesture_recognition_lstm_2p95.bin")
exec_net = iecore.load_network(network=network, device_name="CPU")
input_blob = next(iter(network.input_info))


def test_one():
    data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\pause\pause_0_0.npy")
    data = np.expand_dims(data, axis=0)

    t0 = time.time()
    out = exec_net.infer(inputs={input_blob: data})
    t1 = time.time()
    print(t1 - t0, out)


def run_test_data():
    gestures = ["play", "pause", "forward", "back", "idle", "vol"]
    path = "test_data/test/gesture_landmarks"

    y_true = []
    y_hat = []

    total_time_ns = 0

    for g_index, gesture in enumerate(gestures):
        gesture_path = f"{path}/{gesture}"
        for file in os.listdir(gesture_path):
            file_path = f"{gesture_path}/{file}"
            landmarks = np.load(file_path)

            y_true.append(g_index)

            data = np.expand_dims(landmarks, axis=0).astype(np.float32)

            t0 = time.time_ns()
            result = exec_net.infer(inputs={input_blob: data})
            t1 = time.time_ns()

            total_time_ns += t1 - t0

            y_hat.append(np.argmax(result["result"][0]))

    print("AC on test data", accuracy_score(y_true, y_hat))

    return total_time_ns / (10**9)


def avg_time():
    time = 0

    for i in range(10):
        time += run_test_data()

    print("Average execution time in seconds: ", time / 10)


if __name__ == "__main__":
    # avg_time()
    test_one()