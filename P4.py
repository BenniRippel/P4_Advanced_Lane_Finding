import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

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

    def run(self):
        img = cv2.imread('./test_images/test1.jpg')
        warp = self.warp_image(img)
        self.threshold_image(warp, plot=True)

    def threshold_image(self, image, plot=False):
        """Apply different thresholds to the input image to extract lane lines"""
        # Choose a Sobel kernel size
        ksize = 31  # Choose a larger odd number to smooth gradient measurements
        k_mag = 31
        k_dir = 15

        channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        grady = self.abs_sobel_thresh(channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = self.mag_thresh(channel, sobel_kernel=k_mag, mag_thresh=(30, 100))
        dir_binary = self.dir_threshold(channel, sobel_kernel=k_dir, thresh=(0.7, 1.2))

        # Combine thresholds
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        #combined[(mag_binary == 1) & (dir_binary == 1)] = 1
        #TODO: plot alle 4 threshs in einzelne figures zum tunen
        # kernel = np.ones((3,3), np.uint8)
        # combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        if plot:
            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(combined, cmap='gray')
            ax2.set_title('Thresholded ', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig(self.results + 'test_threshold.png', bbox_inches='tight')
            plt.show()

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

    def mag_thresh(self, channel, sobel_kernel=3, mag_thresh=(0, 255)):
        """Apply 'Magnitude of the Gradient' threshold to an image channel """
        sob_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sob_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        mag = (sob_x ** 2 + sob_y ** 2) ** 0.5
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * mag / np.max(mag))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled < max(mag_thresh)) & (scaled > min(mag_thresh))] = 1
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
        """Undistort and warp image to birds eye view"""
        img_size = (img.shape[1], img.shape[0])    # get image size
        if not self.M:   # check if transformation matrices are defined, if not: get them
            source_img = self.undistort_image(cv2.imread('./test_images/straight_lines2.jpg'))
            self.get_transform_matrices(source_img)
        # undistort and warp the image
        warped = cv2.warpPerspective(self.undistort_image(img), self.M, img_size)

        if plot:
            # plot warped and original image and save it to result_folder
            fig = plt.figure()
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

    P4(result_fldr='./output_images/').run()
    #P4(result_fldr='./output_images/').warp_image(plot=True)

# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
