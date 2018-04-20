import cv2
import numpy as np
from camera_calibration import CameraCalibration
import image_processing as imgproc
from line import Line

scaled_size = 1
image_size = (int(1280 * scaled_size), int(720 * scaled_size))
offset = image_size[1] * 0.3
perspective_src_points = scaled_size * np.float32(
    [[233, 694], [595, 450], [686, 450], [1073, 694]])  # These points are manually selected
perspective_dst_points = np.float32([[offset, image_size[1]], [offset, 0],
                                     [image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]]])
lines = Line(image_size)
# calibration
calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()
print('Camera intrinsix matrix is :')
print(intrinsic_mat)
print('Camera distortion parameters are :')
print(dist_paras)
print()

cap = cv2.VideoCapture('./project_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # distortion correction
    undist = calibration.distort_correction(frame)

    undist = cv2.resize(undist, (0, 0), fx=scaled_size, fy=scaled_size)

    # binary images from different processing methods
    sobel_x_bin = imgproc.abs_sobel_thresh(undist, sobel_kernel=5, orient='x', thresh_min=20, thresh_max=100)
    s_channel_bin = imgproc.hsv_s_threshold(undist, thresh=(80, 255))
    v_channel_bin = imgproc.hsv_v_threshold(undist, thresh=(80, 255))

    # combination of binary images
    thresholded = np.zeros_like(undist[:, :, 0])
    thresholded[(v_channel_bin == 255) & ((sobel_x_bin == 255) | (s_channel_bin == 255))] = 255

    bird_view_img_binary = imgproc.perspective_transfrom(thresholded, perspective_src_points, perspective_dst_points)

    lines.find_line(bird_view_img_binary)

    # visualization
    small_rgb = cv2.resize(undist, (0, 0), fx=0.3, fy=0.3)
    small_thresholded = cv2.resize(thresholded, (0, 0), fx=0.3, fy=0.3)
    small_bird = cv2.resize(bird_view_img_binary, (0, 0), fx=0.3, fy=0.3)
    concatinate_row1 = np.concatenate((small_rgb, cv2.cvtColor(small_thresholded, cv2.COLOR_GRAY2BGR)), axis=1)
    concatinate_row2 = np.concatenate((cv2.cvtColor(small_bird, cv2.COLOR_GRAY2BGR), np.zeros_like(small_rgb)), axis=1)
    concatinate = np.concatenate((concatinate_row1, concatinate_row2), axis=0)
    cv2.imshow('frame', concatinate)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
