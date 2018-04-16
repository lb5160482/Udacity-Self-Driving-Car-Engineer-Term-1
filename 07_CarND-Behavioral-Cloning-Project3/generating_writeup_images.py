import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# data_path = '../input/'
#
# lines = []
# with open(data_path + 'driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     ind = 0
#     for line in reader:
#         lines.append(line)
#         ind += 1
#         if line == 2:
#             break
#
# center_file_path = data_path + 'IMG/' + line[0].split('/')[-1]
# left_file_path = data_path + 'IMG/' + line[1].split('/')[-1]
# right_file_path = data_path + 'IMG/' + line[2].split('/')[-1]
# center_img = cv2.imread(center_file_path)
# left_img = cv2.imread(left_file_path)
# right_img = cv2.imread(right_file_path)
# res = np.concatenate((left_img, center_img), axis=1)
# res = np.concatenate((res, right_img), axis=1)
# res = res[50:140,:,:]
# cv2.imwrite("three_images_cropped.png", res)
#
# print(res.shape)
# cv2.imshow('res', res)
# cv2.waitKey(0)

img = cv2.imread('lane.png')


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_abs = np.abs(sobel)
    scaled_sobel = 250 * sobel_abs / np.max(sobel_abs)
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))
    scaled = 255 * mag / np.max(mag)
    binary = np.zeros_like(scaled)
    binary[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1

    return binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    direction = np.arctan2(sobel_y, sobel_x)
    binary = np.zeros_like(direction)
    binary[(direction > thresh[0]) & (direction < thresh[1])] = 1

    return binary

ksize = 5 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(50, 150))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(50, 150))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(gradx)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255

cv2.imshow('combined', combined)
cv2.waitKey(0)