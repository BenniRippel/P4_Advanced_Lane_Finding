import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from moviepy.editor import VideoFileClip


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

    def run_video(self, video='./project_video.mp4'):
        out_file = video[:-4]+'_output.mp4'   # output file

        clip = VideoFileClip(video)   #read video
        output=clip.fl_image(self.process_frame)   # proces video; function expects color images
        output.write_videofile(out_file, audio=False)    # write video


    def run_image(self, image='./test_images/test1.jpg', save = False):
        img = cv2.imread(image)    # read image
        warp = self.warp_image(img)   # calibrate, undistort and warp
        warped_bin = self.threshold_image(warp, plot=True, save=save)    # threshold image
        left_fit, right_fit = self.slidWin(warped_bin, plot=True, save=save)    # find lines in thresholded img
        self.draw_warped(warp, img, left_fit, right_fit, save=save)    # draw lines
        plt.show()


    def process_frame(self, RGB_frame):
        img = cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR) # convert to bgr
        warp = self.warp_image(img)   # calibrate, undistort and warp
        warped_bin = self.threshold_image(warp)    # threshold image
        left_fit, right_fit = self.slidWin(warped_bin)    # find lines in thresholded img
        result = self.draw_warped(warp, img, left_fit, right_fit)    # draw lines

        return result

    def draw_warped(self, warped, image, left_fit, right_fit, save=False):
        """Draw found lane lines in RGB Image"""
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)
        print(color_warp.shape)

        # define ploty, left_fitx, right_fit_x
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.undistort_image(image), 1, newwarp, 0.3, 0)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8,6), dpi=300)
        plt.imshow(result)
        if save:
            plt.savefig(self.results + 'result.png', bbox_inches='tight')
        return result

    def slidWin(self, binary_warped, margin=100, minpix=50, plot=False, save=False):
        """Udacity implementation of sliding Windows and fit polynomial"""

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
        # Set the width of the windows +/- margin
        # margin = 100
        # Set minimum number of pixels found to recenter window
        # minpix = 50
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
            # If you found > minpix pixels, recenter next window on their mean position
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
            plt.title('Stacked Threshold\n S & L-Channel Color Threshold (green)\n and Gradient Threshold (blue)',
                      fontsize=10)
            fig.add_subplot(1, 2, 2)
            plt.imshow(ret*255, cmap='gray')
            plt.title('Combined Threshold ', fontsize=10)
            if save:
                plt.savefig(self.results + 'test_threshold.png', bbox_inches='tight', orientation='landscape')
        return ret

    def grad_thresh(self, image, chan=1):
        if chan==1:
            m_low = 30
        else:
            m_low = 20
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,chan]
        # Apply each of the thresholding functions
        th_x=self.abs_sobel_thresh(channel, orient='x', sobel_kernel=7, thresh=(30, 255))
        th_y=self.abs_sobel_thresh(channel, orient='y', sobel_kernel=7, thresh=(35, 255))
        th_m=self.mag_thresh(channel, sobel_kernel=5, thresh=(m_low, 255))
        th_d=self.dir_threshold(channel, sobel_kernel=7, thresh=(-0.56, 1.3))
        ret = np.zeros_like(channel)
        ret[((th_m == 1) & (th_d == 1)) | ((th_x == 1) & (th_y == 1)) | (self.combine_color(image) == 1)] = 1
        return ret

    def select_color(self, channel, lower, upper):
        """Color thresholding"""
        ret = np.zeros_like(channel)
        ret[(channel<upper) & (channel>lower)] =1
        return ret

    def combine_color(self, image):
        """Combine colorchanel thresholds"""
        white = self.select_color(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1], 200, 255)
        yellow = self.select_color(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,2], 185, 255)
        # Combine thresholds
        combined = np.zeros_like(white)
        combined[((white == 1) | (yellow == 1))] = 1
        return combined

    def abs_sobel_thresh(self, channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Apply 'Sobel Gradient' threshold to an image channel """
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the derivative or gradient
        abs = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled = np.uint8(255 * abs / np.max(abs))

        binary_output = np.zeros_like(scaled)
        binary_output[(scaled >= min(thresh)) & (scaled <= max(thresh))] = 1
        return binary_output

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
            self.M, self.Minv = SourcePoints.Sourcepoints(image).get_transformation_matrix()
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
    #import argparse
    #parser = argparse.ArgumentParser(description='Calibrates a camera')
    #parser.add_argument('folder_images', type=str, help='The folder containing the calibration images')
    #parser.add_argument('results_folder', type=str, help='The folder where the results are saved')
    #args = parser.parse_args()
    ## assigns the command line argument (usually the video file) to video_src
    #images_folder = args.folder_images
    #results_folder = args.results_folder

    P4(result_fldr='./output_images/').run_video()
    #P4(result_fldr='./output_images/').warp_image(plot=True)

# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
