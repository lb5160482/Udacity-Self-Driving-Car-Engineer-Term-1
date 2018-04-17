import cv2
import glob
from camera_calibration import CameraCalibration
import image_processing as imgproc
import numpy as np

# Part 1 -- Camera calibration
calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()
print('Camera intrinsix matrix is :')
print(intrinsic_mat)
print('Camera distortion parameters are :')
print(dist_paras)
print()

# Part 2 -- Apply a distortion correction to raw images.
calibration.undistort_images()
print()

# Part 3 -- Use color transforms, gradients, etc., to create a thresholded binary image
test_image_paths = glob.glob('./test_images/*.jpg')
print('Start generating binary images from sample images...')
bin_img_dict = {}
for test_img_path in test_image_paths:
	test_img = cv2.imread(test_img_path)
	# test_img = cv2.GaussianBlur(test_img, (5, 5), 0)
	kernel = np.ones((3, 3), np.uint8)
	test_img = cv2.erode(test_img, kernel, iterations=1)
	test_img = cv2.dilate(test_img, kernel, iterations=1)

	# binary images from different processing methods
	sobel_x_bin = imgproc.abs_sobel_thresh(test_img, orient='x',thresh_min=20, thresh_max=100)
	sobel_y_bin = imgproc.abs_sobel_thresh(test_img, orient='y', thresh_min=20, thresh_max=100)
	mag_bin = imgproc.mag_thresh(test_img, sobel_kernel=5, thresh=(30, 100))
	dir_bin = imgproc.dir_threshold(test_img, sobel_kernel=5, thresh=(0.7, 1.3))
	s_channel_bin = imgproc.hls_s_threshol(test_img, thresh=(160, 255))

	# combination of binary images
	combine = np.zeros_like(test_img[:, :, 0])
	combine[((sobel_x_bin == 255) & (sobel_y_bin == 255)) | ((mag_bin == 255) & (dir_bin == 255)) | (
			s_channel_bin == 255)] = 255
	window_name = test_img_path[test_img_path.rfind('\\') + 1:]
	bin_img_dict[window_name] = combine

	# file_name = './output_images/bin_' + window_name;
	# cv2.imwrite(file_name, combine)
print('Finish generating binary images from sample images!')
print()
