import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from moviepy.editor import VideoFileClip
from collections import deque


import Calibration
import SourcePoints

class P4:

    def __init__(self, result_fldr):
        # result folder
        self.results = result_fldr
        # calibration data
        self.cam_matrix = []
        self.dist_coefs = []
        # transformation matrices
        self.M=[]
        self.Minv=[]
        self.y_factor = 0.63
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # deques for temporal information
        self.left_fit = deque(maxlen=10)
        self.right_fit = deque(maxlen=10)
        self.radius = deque(maxlen=10)
        self.offset = deque(maxlen=10)
        self.valid = deque(maxlen=5)
        # data for current frame
        self.current_left_fit_px = []
        self.current_right_fit_px = []
        self.current_left_fit_m = []
        self.current_right_fit_m = []
        self.left_curverad = []
        self.right_curverad = []


    def run_video(self, video='./project_video.mp4'):
        """Run the Lane Finding Pipeline on a input video"""
        out_file = video[:-4]+'_output.mp4'   # output file

        clip = VideoFileClip(video)   #read video
        output=clip.fl_image(self.process_frame)   # process video; function expects color images

        output.write_videofile(out_file, audio=False)    # write video

    def run_image(self, image='./test_images/test5.jpg', save = False):
        """Run the Lane Finding Pipeline on a single input image"""
        img = cv2.imread(image)    # read image
        warp = self.warp_image(img)   # calibrate, undistort and warp
        warped_bin = self.threshold_image(warp, plot=True, save=save)    # threshold image
        self.fit_lines(warped_bin, plot=True, save=save)    # find lines in thresholded img
        self.draw_warped(warp, img, save=save)    # draw lines
        plt.show()

    def process_frame(self, RGB_frame):
        """Frame processing pipeline"""
        img = cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR) # convert to bgr
        warp = self.warp_image(img)   # calibrate, undistort and warp
        warped_bin = self.threshold_image(warp)    # threshold image
        self.fit_lines(warped_bin)    # find lines in thresholded img
        result = self.draw_warped(warp, img)    # draw lines
        result = self.write_to_frame(result)
        result = self.add_info_img(result, warped_bin)
        return result

    def add_info_img(self, frame, img):
        """Add the img to the upper right corner of the frame, scale by 1/3"""
        h, w, c = frame.shape
        h_n = int(h/3)
        w_n = int(w/3)
        img = np.dstack((img*255, img*255, img*255))
        try:
            # plot polynomials
            left_fit = np.mean(self.left_fit, axis=0)
            right_fit = np.mean(self.right_fit, axis=0)
            ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            pts_l = np.array([[a,b] for a,b in zip(left_fitx, ploty)], np.int32)
            pts_r = np.array([[a,b] for a,b in zip(right_fitx, ploty)], np.int32)
            pts_l = pts_l.reshape((-1, 1, 2))
            pts_r = pts_r.reshape((-1, 1, 2))
            cv2.polylines(img, [pts_l], False, (0, 0, 255), 5)
            cv2.polylines(img, [pts_r], False, (255, 0, 0), 5)
        except:
            pass
        img = cv2.resize(img, (w_n, h_n), interpolation = cv2.INTER_AREA)
        frame[:h_n, -w_n:, :] = img
        return frame

    def write_to_frame(self, frame):
        """Write test to frame"""
        cv2.putText(frame, "Radius: %0.0f m" % (np.mean(self.radius)),
                    tuple([50, 50]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Offset: %0.0f cm" % (np.mean(self.offset)),
                    tuple([50, 80]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def calc_radius_and_offset(self, frame):
        """Radius of curvature taken from udacity course"""

        eval = frame.shape[0]-1

        # radius
        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2 * self.current_left_fit_m[0] * eval * self.ym_per_pix +
                                    self.current_left_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * self.current_left_fit_m[0])
        self.right_curverad = ((1 + (2 * self.current_right_fit_m[0] * eval * self.ym_per_pix +
                                     self.current_right_fit_m[1]) ** 2) ** 1.5) / np.absolute(2 * self.current_right_fit_m[0])
        self.radius.append((self.left_curverad + self.right_curverad) * 0.5)

        # offset
        left_fit_bottom = self.current_left_fit_px[0] * eval ** 2 + self.current_left_fit_px[1] * eval + \
                          self.current_left_fit_px[2]
        right_fit_bottom = self.current_right_fit_px[0] * eval ** 2 + self.current_right_fit_px[1] * eval + \
                           self.current_right_fit_px[2]
        self.offset.append(((right_fit_bottom+left_fit_bottom)*0.5 - frame.shape[1]*0.5)*self.xm_per_pix*100)

    def draw_warped(self, warped, image, save=False):
        """Draw found lane lines in RGB Image"""
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)

        # define ploty, left_fitx, right_fit_x
        try:
            left_fit=np.mean(self.left_fit, axis=0)
            right_fit=np.mean(self.right_fit, axis=0)
            ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (255, 102, 102))
            # add left/right and center line
            mean_fit = np.mean([left_fitx, right_fitx], axis=0)
            pts_l = np.array([[a,b] for a,b in zip(left_fitx, ploty)], np.int32)
            pts_r = np.array([[a,b] for a,b in zip(right_fitx, ploty)], np.int32)
            pts_m = np.array([[a,b] for a,b in zip(mean_fit, ploty)], np.int32)
            pts_l = pts_l.reshape((-1, 1, 2))
            pts_r = pts_r.reshape((-1, 1, 2))
            pts_m = pts_m.reshape((-1, 1, 2))
            cv2.polylines(color_warp, [pts_l], False, (0, 255, 0), 20)
            cv2.polylines(color_warp, [pts_r], False, (0, 255, 0), 20)
            cv2.polylines(color_warp, [pts_m], False, (255, 255, 0), 5)

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (image.shape[1], image.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(self.undistort_image(image), 1, newwarp, 0.4, 0)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8,6), dpi=300)
            plt.imshow(result)
            if save:
                plt.savefig(self.results + 'result.png', bbox_inches='tight')
        except:
            print('Could not draw lane lines...')
            result = color_warp
        return result

    def fit_lines(self, binary_warped, margin=50, minpix=50, plot=False, save=False):
        """Udacity implementation of sliding Windows and fit polynomial"""

        # get left and right fit (find new lines if no good fit occured for several frames as well as ate the first frame)
        if not np.any(self.valid):
            self.current_left_fit_px, self.current_right_fit_px = self.find_new_lines(binary_warped, margin=margin,
                                                                                      minpix=minpix)
        else:
            self.current_left_fit_px, self.current_right_fit_px = self.find_line_by_margin(binary_warped, margin=margin)
        # valid_line check
        # check for approx same curvature, for right horizontal distance and  approx parallel

        #TODO: checks obs passt, wenn ja append True zu valid und den fit zu left/right fit
        # append to deques
        if self.lane_dist_check() and self.lane_parallel_check(binary_warped):
            self.left_fit.append(self.current_left_fit_px)
            self.right_fit.append(self.current_right_fit_px)
            self.valid.append(True)
            self.calc_radius_and_offset(binary_warped)
            self.radius.append((self.left_curverad + self.right_curverad) * 0.5)

        else:
            self.valid.append(False)

    def lane_dist_check(self, upper=5.5, lower=2.5):
        """Checks if the distance between the lane lines is between lower and upper threshold in meters"""
        ran=np.arange(1,21)   # range from 1 - 20 m
        left = self.current_left_fit_m[0] * ran ** 2 + self.current_left_fit_m[1] * ran + self.current_left_fit_m[2]
        right = self.current_right_fit_m[0] * ran ** 2 + self.current_right_fit_m[1] * ran + self.current_right_fit_m[2]
        diff = np.absolute(right-left)
        if np.any(diff>upper) or np.any(diff<lower): return False
        return True

    def lane_parallel_check(self, frame,  thresh = 1):
        "Checks if the lines are parallel at different locations "
        ran=np.arange(1,21)   # range from 1 - 20 m
        left = self.current_left_fit_m[0] * ran ** 2 + self.current_left_fit_m[1] * ran + self.current_left_fit_m[2]
        right = self.current_right_fit_m[0] * ran ** 2 + self.current_right_fit_m[1] * ran + self.current_right_fit_m[2]
        diff = (right-left)-np.mean(right-left)
        if np.any(diff>thresh): return False
        return True

    def find_new_lines(self, binary_warped, margin=100, minpix=50, plot=False, save=False):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their median position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.median(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.median(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit new polynomials to x,y in world space
        self.current_left_fit_m = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        self.current_right_fit_m = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)


        if plot:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.figure(figsize=(8,6), dpi=300)
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            if save:
                plt.savefig(self.results + 'test_lane_detection.png', bbox_inches='tight')
        return left_fit, right_fit

    def find_line_by_margin(self, binary_warped, margin=100):
        """ Find lines based on previously found lane locations. Taken from udacity course"""
        left_fit=np.mean(self.left_fit, axis=0)
        right_fit=np.mean(self.right_fit, axis=0)


        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit new polynomials to x,y in world space
        self.current_left_fit_m = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        self.current_right_fit_m = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        return left_fit, right_fit

    def threshold_image(self, image, plot=False, save=False):
        """Apply different thresholds to the input image to extract lane lines"""

        grad_1 = self.grad_thresh(image, chan=1)
        grad_2 = self.grad_thresh(image, chan=2)

        grad = np.zeros_like(grad_1)
        grad[(grad_1 == 1) | (grad_2 == 1)]=1
        ret = np.zeros_like(grad_1)
        ret[(grad == 1) | (self.combine_color(image) == 1)] = 1

        # Plot stacked thresholds
        if plot:
            # Plot the result
            color = np.zeros_like(grad)
            color[(self.combine_color(image) == 1)] = 1
            stacked = np.dstack((np.zeros_like(grad), color, grad))
            fig=plt.figure(figsize=(8,6), dpi=300)
            fig.add_subplot(1, 2, 1)
            plt.imshow(stacked*255)
            plt.title('Stacked Threshold\n Color Threshold (green)\n and Gradient Threshold (blue)',
                      fontsize=10)
            fig.add_subplot(1, 2, 2)
            plt.imshow(ret*255, cmap='gray')
            plt.title('Combined Threshold ', fontsize=10)
            if save:
                plt.savefig(self.results + 'test_threshold.png', bbox_inches='tight', orientation='landscape')
        return ret

    def grad_thresh(self, image, chan=1):
        if len(image.shape) == 3:
            if chan==1:
                m_low = 30
            else:
                m_low = 20
            channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,chan]
        else:
            channel = image
        # Apply each of the thresholding functions
        th_m=self.mag_thresh(channel, sobel_kernel=5, thresh=(m_low, 255))
        th_d=self.dir_threshold(channel, sobel_kernel=7, thresh=(-0.56, 1.3))
        ret = np.zeros_like(channel)
        ret[((th_m == 1) & (th_d == 1)) | (self.combine_color(image) == 1)] = 1

        return ret

    def select_color(self, channel, lower, upper):
        """Color thresholding"""
        ret = np.zeros_like(channel)
        ret[(channel<upper) & (channel>lower)] =1
        return ret

    def combine_color(self, image):
        """Combine colorchanel thresholds"""

        l = self.select_color(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1], 205, 245)
        s = self.select_color(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,2], 100, 180)

        # Combine thresholds
        combined = np.zeros_like(s)
        combined[((l == 1) | (s == 1))] = 1
        return combined

    def mag_thresh(self, channel, sobel_kernel=3, thresh=(0, 255)):
        """Apply 'Magnitude of the Gradient' threshold to an image channel """
        sob_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sob_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        mag = (sob_x ** 2 + sob_y ** 2) ** 0.5
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * mag / np.max(mag))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled < max(thresh)) & (scaled > min(thresh))] = 1
        return binary_output

    def dir_threshold(self, channel, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """Apply 'Direction of the Gradient' threshold to an image channel """
        # Take the absolute value of the x and y gradients
        abs_x = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_y = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        di = np.arctan2(abs_y, abs_x)
        # Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(abs_x)
        binary_output[(di > min(thresh)) & (di < max(thresh))] = 1
        return binary_output

    def warp_image(self, img=cv2.imread('./test_images/test5.jpg'), plot=False):
        """Undistort and warp image to birds eye view, expects bgr images"""
        img_size = (img.shape[1], img.shape[0])    # get image size
        if self.M == []:   # check if transformation matrices are defined, if not: get them
            source_img = self.undistort_image(cv2.imread('./test_images/straight_lines2.jpg'))
            self.get_transform_matrices(source_img)
        # undistort and warp the image
        warped = cv2.warpPerspective(self.undistort_image(img), self.M, img_size)

        if plot:
            # plot warped and original image and save it to result_folder
            fig = plt.figure(figsize=(8,6), dpi=300)
            fig.add_subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Undistorted Image')
            fig.add_subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title('Warped Image')
            plt.savefig(self.results + 'test_warped.png', bbox_inches='tight')
        return warped

    def get_transform_matrices(self, image):
        """Get the source points for warping images to bird-eye view"""
        try:    # load image
            self.M, self.Minv = SourcePoints.Sourcepoints(image, min_y_factor=self.y_factor).get_transformation_matrix()
        except:
            print('Failed to get the transformation matrices for image warping!')

    def test_undistortion(self, file='./camera_cal/calibration1.jpg'):
        """Test the calibration by undistorting and plotting an image!"""
        try:
            if not self.cam_matrix:    # check for calib data, get it if not already loaded
                self.get_calibration_data()
            img = cv2.imread(file)    # load image as grayscale
            und_img = self.undistort_image(img)    # undistort
            # plot distorted and undistorted image and save it to result_folder
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('Original Image')
            fig.add_subplot(1, 2, 2)
            plt.imshow(und_img)
            plt.title('Undistorted Image')
            plt.savefig(self.results+'test_undistortion.png', bbox_inches='tight')
        except:
            print('Something went horribly wrong!')

    def undistort_image(self, img):
        """
        Undistorts an image using self.dist_coefs
        :param img: distorted image
        :return: undistorted image
        """
        if not list(self.cam_matrix):
            self.get_calibration_data()
        return cv2.undistort(img, self.cam_matrix, self.dist_coefs)

    def get_calibration_data(self):
        """Tries to load calibration data from './calibration_parameters/'. If this fails, the camera is calibrated with
         images from './camera_cal/. Throws an error if both approaches fail'"""

        try:
            self.cam_matrix = np.load('./calibration_parameters/Cameramatrix.npy')
            self.dist_coefs = np.load('./calibration_parameters/DistortionCoeffs.npy')
        except:
            print("Couldn't load calibration data. Starting calibration...")
            try:
                self.cam_matrix, self.dist_coefs = Calibration.Calibration('./camera_cal/', save=True).run()
                print('...done!')
            except:
                print('Calibration failed. Aborting...')



def main():

    P4(result_fldr='./output_images/').run_video()
    #P4(result_fldr='./output_images/').run_image()

# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
