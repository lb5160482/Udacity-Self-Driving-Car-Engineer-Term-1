import cv2
from Vehicle_Detector import Vehicle_Detector
import os
import pickle


cap = cv2.VideoCapture('./project_video.mp4')

file_name = 'model_dict'
if os.path.exists(file_name):
    file_object = open(file_name, 'rb')
    model_dict = pickle.load(file_object)
else:
    raise FileNotFoundError('model_dict not found!')
test_file = './test_images/test1.jpg'
test_image = cv2.imread(test_file)
img_shape = test_image.shape
windows = [64]
vehicle_detector = Vehicle_Detector(model_dict, img_shape, windows)

while cap.isOpened():
    ret, frame = cap.read()
    vehicle_detector.feed(frame)