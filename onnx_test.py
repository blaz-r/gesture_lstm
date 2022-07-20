import numpy as np
import onnx, onnxruntime
import time
import os

from sklearn.metrics import accuracy_score

model = "gesture_recognition_lstm_2p95.onnx"
session = onnxruntime.InferenceSession(model, None)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(input_name)
print(output_name)


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
            result = session.run([output_name], {input_name: data})
            t1 = time.time_ns()

            total_time_ns += t1 - t0

            y_hat.append(np.argmax(result[0][0]))

    print("AC on test data", accuracy_score(y_true, y_hat))

    return total_time_ns / (10**9)


def run_one():
    data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\pause\pause_0_0.npy").astype(np.float32)
    data = np.expand_dims(data, axis=0)

    t0 = time.time_ns()
    result = session.run([output_name], {input_name: data})
    t1 = time.time_ns()
    print(t1 - t0, result)


def avg_time():
    time = 0

    for i in range(10):
        time += run_test_data()

    print("Average execution time in seconds: ", time / 10)


if __name__ == "__main__":
    avg_time()
