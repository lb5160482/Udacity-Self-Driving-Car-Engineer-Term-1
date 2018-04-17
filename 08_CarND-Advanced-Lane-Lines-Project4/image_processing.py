import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if (orient == 'x'):
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	else:
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	sobel_absolute = np.absolute(sobel)
	scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))
	binary = np.zeros_like(scaled)
	binary[(scaled >= thresh_min) & (scaled <= thresh_max)] = 255

	return binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	sobel_mag = np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))
	scaled_mag = 255 * sobel_mag / np.max(sobel_mag)
	binary_output = np.zeros_like(scaled_mag)
	binary_output[(scaled_mag > thresh[0]) & (scaled_mag < thresh[1])] = 255

	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
	sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
	direction = np.arctan2(sobel_y, sobel_x)
	threshed_direction = np.zeros_like(direction)
	threshed_direction[(direction > thresh[0]) & (direction < thresh[1])] = 255

	return threshed_direction


def hls_s_threshol(img, thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	s_channel = hls[:, :, 2]
	binary = np.zeros_like(s_channel)
	binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

	return binary