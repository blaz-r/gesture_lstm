import numpy as np
from openvino.runtime import Core
from openvino.inference_engine import IECore
import time


data = np.load("G:\Faks\diploma\gesture_capture\gesture_landmarks\play\play_1.npy").astype(np.float32)
data = np.expand_dims(data, axis=0)

# ie = Core()
# model = ie.read_model(model="gesture_recognition_lstm.xml", weights="gesture_recognition_lstm.bin")
# compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
#
# input_key = compiled_model.input(0)
# output_key = compiled_model.output(0)
# network_input_shape = input_key.shape
#
# result = compiled_model([data])[output_key]
# print(result)

iecore = IECore()

network = iecore.read_network(model="gesture_recognition_lstm.xml", weights="gesture_recognition_lstm.bin")
exec_net = iecore.load_network(network=network, device_name="CPU")
input_blob = next(iter(network.input_info))
t0 = time.time()
out = exec_net.infer(inputs={input_blob: data})
t1 = time.time()
print(t1 - t0, out)
