from math import gcd

import cv2
import mediapipe as mp
import depthai as dai

def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height
        other = width
    else:
        reference = width
        other = height
    size_candidates = {}
    for s in range(288, reference, 16):
        f = gcd(reference, s)
        n = s // f
        d = reference // f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)

    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]


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


def create_pipeline():
    pipeline = dai.Pipeline()

    print("Adding RGB camera")
    rgb_cam = pipeline.createColorCamera()
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.setInterleaved(False)
    rgb_cam.setFps(30)
    rgb_cam.setIspScale(scale_nd[0], scale_nd[1])
    rgb_cam.setVideoSize(img_w, img_h)
    rgb_cam.setPreviewSize(img_w, img_h)

    print("Adding video out to queue")
    cam_out = pipeline.createXLinkOut()
    cam_out.setStreamName("cam_out")
    cam_out.input.setQueueSize(1)
    cam_out.input.setBlocking(False)
    rgb_cam.video.link(cam_out.input)

    return pipeline


def capture():
    device = dai.Device()
    device.startPipeline(create_pipeline())
    img_q = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)

    for cap in range(30):

        image = img_q.get().getCvFrame()
        cv2.putText(image, f"Starting collection in 3 seconds", (120,200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 120, 255), 4, cv2.LINE_AA)
        cv2.imshow("Capture", image)
        key = cv2.waitKey(3000)
        if key == 27 or key == ord('q'):
            break

        for frame in range(30):
            # 1152 x 648
            # out = cv2.VideoWriter(f"gesture_videos/play/play_capture{cap}.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            #                       30, (img_w, img_h))

            image = img_q.get().getCvFrame()

            # out.write(image)

            cv2.putText(image, f"Capturing video {cap}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Capture", image)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

            # out.release()

    cv2.destroyAllWindows()


def read_test():
    cap = cv2.VideoCapture("G:\Vids\Davinci\geste_test\CutTest.mp4")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(f"{frame_width} x {frame_height}")

    progress = 0
    while cap.isOpened():
        ret, frame = cap.read()

        progress += 1
        print(f"Progress: {progress}/{frame_count}")

        if ret:
            cv2.imshow("Read", frame)
            key = cv2.waitKey(20)
        else:
            break

    cap.release()


if __name__ == "__main__":
    capture()
    # read_test()
