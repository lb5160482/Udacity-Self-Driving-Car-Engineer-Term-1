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
        self.pix_per_cell = model_dict['pix_per_cel']
        self.cell_per_block = model_dict['cell_per_block']
        self.spatial_size = model_dict['spatial_size']
        self.hist_bins = model_dict['hist_bins']
        self.color_space = model_dict['color_space']
        self.hog_channel = model_dict['hog_channel']
        self.spatial_feat = model_dict['spatial_feat']
        self.hist_feat = model_dict['hist_feat']
        self.hog_feat = model_dict['hog_feat']
        self.img_shape = img_shape
        self.windows = windows
        self.y_start = img_shape[0] // 2
        self.y_stop = img_shape[0]
        self.cur_rects = []
        self.scale = 1.5

    def feed(self, img):
        img_to_search = img[self.y_start:self.y_stop, :, :]
        self.cur_rects = self.get_cur_rects(img_to_search)

    def get_cur_rects(self, img):
        draw_img = np.copy(img)
        imshape = img.shape
        if self.scale != 1:
            img = cv2.resize(img, (np.int(imshape[1] / self.scale), np.int(imshape[0] / self.scale)))
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

        ch1 = feature_image[:, :, 0]
        ch2 = feature_image[:, :, 1]
        ch3 = feature_image[:, :, 2]

        # compute steps
        nx_blocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        ny_blocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeature_per_block = self.orient * self.cell_per_block ** 2
        for window in self.windows:
            nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
            cells_per_step = int(window / self.pix_per_cell / 4)
            nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step + 1
            ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step + 1
            if self.hog_channel == 'ALL':
                hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
                hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
                hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            else:
                hog = self.get_hog_features(feature_image[:, :, self.hog_channel], self.orient, self.pix_per_cell,
                                            self.cell_per_block, feature_vec=False)
            for xb in range(nx_steps):
                for yb in range(ny_steps):
                    ypos = yb * self.cell_per_block
                    xpos = xb * self.cell_per_block
                    img_features = []

                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    subimg = cv2.resize(feature_image[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # get color features
                    if self.spatial_feat is True:
                        spatial_features = self.bin_spatial(subimg, size=self.spatial_size)
                        img_features.append(spatial_features)
                    if self.hist_feat is True:
                        hist_features = self.color_hist(subimg, nbins=self.hist_bins)
                        img_features.append(hist_features)
                    # get hog features
                    if self.hog_feat is True:
                        if self.hog_channel == 'ALL':
                            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                        else:
                            hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        img_features.append(hog_features)
                    img_features = np.concatenate(img_features)

                    test_features = self.X_scalar.transform(img_features.reshape(1, -1))
                    test_prediction = self.svc.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft * self.scale)
                        ytop_draw = np.int(ytop * self.scale)
                        win_draw = np.int(window * self.scale)
                        cv2.rectangle(draw_img, (xbox_left, ytop_draw), (xbox_left + win_draw, ytop_draw + win_draw),
                                      color=(255, 0, 0))

        cv2.imshow('aaa', draw_img)
        cv2.waitKey(1)

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

    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

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
