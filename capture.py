import time
import cv2
import depthai as dai
from dai_utils import create_pipeline, find_isp_scale_params


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


def capture(path, name):
    """
    Capture gesture videos.
    Each call captures 20 videos with 30 frames each.
    All are 30fps with 1152 x 648 resolution

    :param path: path to gesture directory
    :param name: filename of video
    :return: None
    """
    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    image = img_q.get().getCvFrame()

    for cap in range(20):

        last_time = time.time()
        while time.time() - last_time < 2:
            image = img_q.get().getCvFrame()

            cv2.putText(image, f"Starting collection in {int(3 - (time.time() - last_time))} seconds",
                        (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 120, 255), 4, cv2.LINE_AA)
            cv2.imshow("Capture", image)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                return

        # 1152 x 648
        out = cv2.VideoWriter(f"{path}/{name}{cap}.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                              30, (img_w, img_h))

        for frame in range(30):
            image = img_q.get().getCvFrame()

            out.write(image)

            cv2.putText(image, f"Capturing video {cap}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Capture", image)

            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                return

        out.release()

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
    capture("gesture_videos/pause", "pause_capture")
    # read_test("gesture_videos/play/play_capture1.mp4")
