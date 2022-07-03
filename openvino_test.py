import numpy as np
from openvino.inference_engine import IECore


data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\pause\pause_1.npy").astype(np.float16) / 10000
data = np.expand_dims(data, axis=0)

iecore = IECore()

network = iecore.read_network(model="gesture_recognition_lstm.xml", weights="gesture_recognition_lstm.bin")
exec_net = iecore.load_network(network=network, device_name="MYRIAD")
input_blob = next(iter(network.input_info))
out = exec_net.infer(inputs={input_blob: data})
print(out)
