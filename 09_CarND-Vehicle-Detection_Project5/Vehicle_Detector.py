import cv2
import os
import pickle
import platform
import numpy as np
from skimage.feature import hog

class Vehicle_Detector():
	def __init__(self, model_dict, img_shape, windows):
		self.svc = model_dict['svc']
		self.X_scalar = model_dict['scalar']
		self.orient = model_dict['orient']
		self.pix_per_cel = model_dict['pix_per_cel']
		self.cell_per_block = model_dict['cell_per_block']
		self.spatial_size = model_dict['spatial_size']
		self.hist_bins = model_dict['hist_bins']
		self.color_space = model_dict['color_space']
		self.img_shape = img_shape
		self.windows = windows
		self.y_start = img_shape[0] // 2
		self.y_stop = img_shape[0]
		self.cur_rects = []

	def feed(self, img):
		img_to_search = img[self.y_start:self.y_stop, :, :]
		self.cur_rects = self.get_cur_rects(img_to_search)

	def get_cur_rects(self, img):
		if self.color_space != 'BGR':
			if self.color_space == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			elif self.color_space == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
			elif self.color_space == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
			elif self.color_space == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
			elif self.color_space == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
			elif self.color_space == 'RGB':
				feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			else:
				feature_image = np.copy(img)
		else:
			feature_image = np.copy(img)

		ch1 = feature_image[:,:,0]
		ch2 = feature_image[:, :, 1]
		ch3 = feature_image[:, :, 2]

		# compute steps
		nx_blocks = (ch1.shape[1] // self.pix_per_cel) - self.cell_per_block + 1
		ny_blocks = (ch1.shape[0] // self.pix_per_cel) - self.cell_per_block + 1
		nfeature_per_block = self.orient * self.cell_per_block**2
		for window in self.windows:
			nblocks_per_window = (window // self.pix_per_cel) - self.cell_per_block + 1
			cells_per_step = int(window / self.pix_per_cel / 4)
			nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
			ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step + 1
			# nx_steps =


		cv2.imshow('aaa', feature_image)
		cv2.waitKey(0)


	# Define a function to return HOG features and visualization
	def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
						 vis=False, feature_vec=True):
		# Call with two outputs if vis==True
		if vis == True:
			features, hog_image = hog(img, orientations=orient,
									  pixels_per_cell=(pix_per_cell, pix_per_cell),
									  block_norm='L2-Hys',
									  cells_per_block=(cell_per_block, cell_per_block),
									  transform_sqrt=True,
									  visualise=vis, feature_vector=feature_vec)
			return features, hog_image
		# Otherwise call with one output
		else:
			features = hog(img, orientations=orient,
						   pixels_per_cell=(pix_per_cell, pix_per_cell),
						   cells_per_block=(cell_per_block, cell_per_block),
						   block_norm='L2-Hys',
						   transform_sqrt=True,
						   visualise=vis, feature_vector=feature_vec)
			return features


if __name__ == '__main__':
	file_name = 'model_dict'
	if os.path.exists(file_name):
		file_object = open(file_name, 'rb')
		model_dict = pickle.load(file_object)
	else:
		raise FileNotFoundError('model_dict not found!')

	if platform.system() == 'Windows':
		test_file = '.\\test_images\\test1.jpg'
	else:
		test_file = './test_images/test1.jpg'
	test_image = cv2.imread(test_file)
	img_shape = test_image.shape

	windows = [64]
	vehicle_detector = Vehicle_Detector(model_dict, img_shape, windows)

	rects = vehicle_detector.feed(test_image)
