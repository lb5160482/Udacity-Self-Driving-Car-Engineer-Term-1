import csv
import cv2
import numpy as np

data_path = '../input/'

lines = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    ind = 0
    for line in reader:
        lines.append(line)
        ind += 1
        if line == 2:
            break

center_file_path = data_path + 'IMG/' + line[0].split('/')[-1]
left_file_path = data_path + 'IMG/' + line[1].split('/')[-1]
right_file_path = data_path + 'IMG/' + line[2].split('/')[-1]
center_img = cv2.imread(center_file_path)
left_img = cv2.imread(left_file_path)
right_img = cv2.imread(right_file_path)
res = np.concatenate((left_img, center_img), axis=1)
res = np.concatenate((res, right_img), axis=1)
res = res[50:140,:,:]
cv2.imwrite("three_images_cropped.png", res)

print(res.shape)
cv2.imshow('res', res)
cv2.waitKey(0)
