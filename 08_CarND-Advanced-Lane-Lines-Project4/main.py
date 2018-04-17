import cv2
from camera_calibration import CameraCalibration

# Part 1 -- Camera calibration
calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()
print('Camera intrinsix matrix is :')
print(intrinsic_mat)
print('Camera distortion parameters are :')
print(dist_paras)

# Part 2 -- Apply a distortion correction to raw images.
calibration.undistort_images()