import cv2
import numpy as np
from camera_calibration import CameraCalibration
import image_processing as imgproc

image_size = (1280, 720)
offset = 200
perspective_src_points = np.float32([[233, 694], [595, 450], [686, 450], [1073, 694]])  # These points are manually selected
perspective_dst_points = np.float32([[offset, image_size[1]], [offset, 0],
                                     [image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]]])

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
    # kernel = np.ones((3, 3), np.uint8)
    # undist = cv2.erode(undist, kernel, iterations=1)
    # undist = cv2.dilate(undist, kernel, iterations=1)

    # binary images from different processing methods
    sobel_x_bin = imgproc.abs_sobel_thresh(undist, orient='x', thresh_min=20, thresh_max=100)
    sobel_y_bin = imgproc.abs_sobel_thresh(undist, orient='y', thresh_min=20, thresh_max=100)
    # mag_bin = imgproc.mag_thresh(undist, sobel_kernel=5, thresh=(30, 100))
    # dir_bin = imgproc.dir_threshold(undist, sobel_kernel=5, thresh=(0.7, 1.3))
    s_channel_bin = imgproc.hls_s_threshol(undist, thresh=(150, 255))

    # combination of binary images
    thresholded = np.zeros_like(undist[:, :, 0])
    thresholded[((sobel_x_bin == 255) & (sobel_y_bin == 255)) | (s_channel_bin == 255)] = 255

    bird_view_img_binary = imgproc.perspective_transfrom(thresholded, perspective_src_points, perspective_dst_points)

    small_rgb = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    small_thresholded = cv2.resize(thresholded, (0, 0), fx=0.3, fy=0.3)
    small_bird = cv2.resize(bird_view_img_binary, (0, 0), fx=0.3, fy=0.3)
    concatinate = np.concatenate((cv2.cvtColor(small_thresholded, cv2.COLOR_GRAY2BGR), small_rgb, cv2.cvtColor(small_bird, cv2.COLOR_GRAY2BGR)), axis=1)
    cv2.imshow('frame', concatinate)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
