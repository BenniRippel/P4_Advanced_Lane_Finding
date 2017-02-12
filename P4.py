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

    def warp_image(self, img=cv2.imread('./test_images/straight_lines2.jpg'), plot=False):
        img_size = (img.shape[1], img.shape[0])
        if not self.M:
            #TODO: in SourcePoints auch undistorted images nehmen, evtl auch roi etwas kuerzer, auch houghlines verbessern (weniger)
            self.get_transform_matrices()
        warped = cv2.warpPerspective(self.undistort_image(img), self.M, img_size)

        if plot:
            # plot warped and original image and save it to result_folder
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(img,)
            plt.title('Original Image')
            fig.add_subplot(1, 2, 2)
            plt.imshow(warped)
            plt.title('Undistorted Image')
            plt.show()
            plt.savefig(self.results + 'test_warped.png', bbox_inches='tight')

    def get_transform_matrices(self, file='./test_images/straight_lines2.jpg'):
        """Get the source points for warping images to bird-eye view"""
        try:    # load image
            self.M, self.Minv = SourcePoints.Sourcepoints(file).get_transformation_matrix()
        except:
            print('Failed to get the transformation matrices for image warping!')

    def test_undistortion(self, file = './camera_cal/calibration1.jpg'):
        """Test the calibration by undistorting and plotting an image!"""
        try:
            if not self.cam_matrix:    # check for calib data, get it if not already loaded
                self.get_calibration_data()
            img = cv2.imread(file ,0)    # load image as grayscale
            und_img = self.undistort_image(img)    # undistort

            # plot distorted and undistorted image and save it to result_folder
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(img, cmap = 'gray')
            plt.title('Original Image')
            fig.add_subplot(1, 2, 2)
            plt.imshow(und_img, cmap='gray')
            plt.title('Undistorted Image')
            plt.savefig(self.results+'test_undistortion.png', bbox_inches='tight')
        except:
            print('Something went horribly wrong! Check the filepath...')

    def undistort_image(self, img):
        """
        Undistorts an image using self.dist_coefs
        :param img: distoted image
        :return: undistorted image
        """
        if not self.cam_matrix:
            self.get_calibration_data()
        return cv2.undistort(img, self.cam_matrix, self.dist_coefs, None, self.cam_matrix)

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
    P4(result_fldr='./output_images/').warp_image(plot=True)
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
