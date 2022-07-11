import re
import time
import cv2
import depthai as dai
import numpy as np

from dai_utils import create_pipeline, find_isp_scale_params
import mediapipe as mp
from train_model import extract_landmarks

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


def view():
    """
    View video directly from camera

    :return: None
    """

    # create device with pipeline, with same params as final project
    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    # capture and display video
    while True:
        image = img_q.get().getCvFrame()
        cv2.imshow("View", image)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            return


def handle_mediapipe(image, hands):
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

    return results


def capture(gesture_videos_path, gesture_landmarks_path, name, capture_num=20):
    """
    Capture gesture videos.
    Each call captures 20 videos with 30 frames each.
    All are 30fps with 1152 x 648 resolution

    :param gesture_videos_path: path to gesture video directory
    :param gesture_landmarks_path: path to gesture numpy landmarks directory
    :param name: filename
    :param capture_num: number of videos to be captured
    :return: None
    """
    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    image = img_q.get().getCvFrame()

    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.05,
            min_tracking_confidence=0.3) as hands:
        for cap in range(capture_num):

            last_time = time.time()
            while time.time() - last_time < 2:
                image = img_q.get().getCvFrame()

                cv2.putText(image, f"Starting collection in {int(3 - (time.time() - last_time))} seconds",
                            (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 120, 255), 4, cv2.LINE_AA)

                results = handle_mediapipe(image, hands)
                while not results.multi_hand_landmarks:
                    image = img_q.get().getCvFrame()
                    results = handle_mediapipe(image, hands)

                cv2.imshow("Capture", image)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    return

            # 1152 x 648
            out = cv2.VideoWriter(f"{gesture_videos_path}/{name}_{cap}.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  30, (img_w, img_h))

            # 30 frames with 42 landmarks
            landmarks = np.zeros((30, 42))

            frame_idx = 0
            for frame in range(30):
                image = img_q.get().getCvFrame()
                out.write(image)

                cv2.putText(image, f"Capturing video {cap}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

                results = handle_mediapipe(image, hands)
                while not results.multi_hand_landmarks:
                    image = img_q.get().getCvFrame()
                    results = handle_mediapipe(image, hands)

                cv2.imshow("Capture", image)

                # insert scaled landmarks for current frame
                landmarks[frame_idx, :] = extract_landmarks(results)
                frame_idx += 1

                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'):
                    return

            out.release()
            # if while didn't exit due to no landmark detections
            np_path = f"{gesture_landmarks_path}/{name}_{cap}"
            np.save(np_path, landmarks)

    cv2.destroyAllWindows()


def read_test(path):
    """
    Show video from file

    :param path: path to video
    :return: None
    """
    cap = cv2.VideoCapture(path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(f"{frame_width} x {frame_height}")

    progress = 0
    while cap.isOpened():
        ret, frame = cap.read()

        print(f"Progress: {progress}/{frame_count}")
        progress += 1

        if ret:
            cv2.imshow("Read", frame)
            key = cv2.waitKey(20)
        else:
            break

    cap.release()


if __name__ == "__main__":
    view()
    name = "idle"
    lm_path = f"gesture_landmarks/{name}"
    video_path = f"gesture_videos/{name}"
    filename = f"{name}_1"
    capture(video_path, lm_path, filename, capture_num=40)
    # read_test("gesture_videos/play/play_capture1.mp4")
