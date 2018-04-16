import cv2
from camera_calibration import CameraCalibration

# Part 1 -- Camera calibration
calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()

