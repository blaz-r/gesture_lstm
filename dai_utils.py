from math import gcd
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

def create_pipeline():
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