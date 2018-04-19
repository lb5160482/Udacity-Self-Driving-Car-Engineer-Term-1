import numpy as np
import cv2

class Line():
    def __init__(self):
        self.detected = False
        self.recent_xfit = []
        self.best_x = None
        self.best_fit = None
        self.current_fit = None
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = None
        self.allx = None
        self.ally = None

    def find_line(self, input_binary_warped, side):
        if not self.detected:
            self.blind_search(input_binary_warped, side)
        else:
            raise NotImplementedError()


    # Note: binary_warped will be only half on the image
    def blind_search(self, input_binary_warped, side):
        if side is 'left':
            binary_warped = input_binary_warped[:, 0:input_binary_warped.shape[1] // 2]
        elif side is 'right':
            binary_warped = input_binary_warped[:, input_binary_warped.shape[1] // 2:]

        #### visualization ####
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        #### visualization ####

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        x_base = np.argmax(histogram)

        nwindow = 9
        window_height = np.int(binary_warped.shape[0] // nwindow)

        nonezero = binary_warped.nonzero()
        nonezero_y = np.array(nonezero[0])
        nonezero_x = np.array(nonezero[1])


        x_current = x_base
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        lane_inds = []

        for window_ind in range(nwindow):
            win_y_low = binary_warped.shape[0] - (window_ind + 1) * window_height
            win_y_high = binary_warped.shape[0] - window_ind * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            #### visualization ####
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)
            #### visualization ####

            good_indx = ((nonezero_y >= win_y_low) & (nonezero_y <= win_y_high) & (nonezero_x >= win_x_low) & (nonezero_x <= win_x_high)).nonzero()[0]
            lane_inds.append(good_indx)
            if len(good_indx) >minpix:
                x_current = np.int(np.mean(nonezero_x[good_indx]))

        lane_inds = np.concatenate(lane_inds)
        x = nonezero_x[lane_inds]
        y = nonezero_y[lane_inds]

        fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0], dtype=np.int32)
        fitx = (fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]).astype(np.int32)
        pts = np.vstack((fitx, ploty)).T.reshape(-1, 1, 2)
        out_img = cv2.polylines(out_img, [pts], False, (0, 0, 255))
        cv2.imshow('out', out_img)
        cv2.waitKey(0)
