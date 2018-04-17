import cv2
from camera_calibration import CameraCalibration
import image_processing as imgproc
import numpy as np

scale_factor = 1

file_name = './output_images/undist_test1.jpg'

calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()

#######################
img = cv2.imread(file_name)
undist = calibration.distort_correction(img)
undist = cv2.resize(undist, (0, 0), fx=scale_factor, fy=scale_factor)

hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
h = hls[:, :, 0]
s = hls[:, :, 1]
v = hls[:, :, 2]
s_bin = np.zeros_like(s)
s_bin[(s >= 100) & (s <= 255)] = 255
v_bin = np.zeros_like(v)
v_bin[(v >= 80) & (v <= 255)] = 255

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
gray_bin = np.zeros_like(gray)

gray_bin[gray > 70] = 255
sobel_x_bin = imgproc.abs_sobel_thresh(undist, orient='x', sobel_kernel=5, thresh_min=20, thresh_max=100)
sobel_y_bin = imgproc.abs_sobel_thresh(undist, orient='y', thresh_min=20, thresh_max=100)
mag_bin = imgproc.mag_thresh(undist, sobel_kernel=5, thresh=(30, 100))
dir_bin = imgproc.dir_threshold(undist, sobel_kernel=5, thresh=(0.7, 1.3))
s_channel_bin = imgproc.hsv_s_threshold(undist, thresh=(80, 255))

combine = np.zeros_like(sobel_x_bin)
combine[(v_bin == 255) & ((sobel_x_bin == 255) | (s_channel_bin == 255))] = 255

cv2.imshow('img', undist)
cv2.imshow('distortion comparison', cv2.addWeighted(cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor), 0.5,
													undist, 0.5, 0))
cv2.imshow('sobel_x', sobel_x_bin)
cv2.imshow('combine', combine)
cv2.imshow('s', s)
cv2.imshow('s_channel_bin', s_channel_bin)
cv2.imshow('v', v)
cv2.imshow('v_bin', v_bin)
cv2.waitKey(0)

cv2.destroyAllWindows()