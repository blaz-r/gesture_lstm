import numpy as np
import onnx, onnxruntime
import time

data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\pause\pause_1.npy").astype(np.float32)
data = np.expand_dims(data, axis=0)

model = "gesture_recognition_lstm.onnx"

session = onnxruntime.InferenceSession(model, None)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(input_name)
print(output_name)

t0 = time.time()
result = session.run([output_name], {input_name: data})
t1 = time.time()
print(t1 - t0, result)