import numpy as np
import cv2
import os
import glob


class Calibration:
    """Calibrates a camera with chessboard images"""
    def __init__(self, img_fldr, results_fldr='./calibration_parameters/', pattern=(9, 6), save=False, ret=True):
        # save results?
        self.save = save
        # return results?
        self.ret = ret
        # init folders
        self.MonoLeft = img_fldr
        self.results_fldr = results_fldr
        self.check_for_folder(results_fldr)
        # calibration pattern size
        self.pattern_size = pattern
        # PatternPoints
        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        # terminationcriteria for corner_sub_pix
        self.term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 0.001)
        # image width & height
        self.w, self.h = 0, 0
        # calibration parameters
        self.camera_matrix = []
        self.dist_coefs = []
        # calibration error
        self.rms = 0

    def run(self):
        # linke Cam Mono
        img_p, obj_p = self.prep_calib(self.MonoLeft)
        self.rms, self.camera_matrix, self.dist_coefs = self.calibrate(obj_p, img_p)
        self.print_results()
        if self.save:
            self.save_parameters()
        if self.ret:
            return self.camera_matrix, self.dist_coefs

    def calibrate(self, obj_poi, img_poi):
        """ Calibrates a camera
        :param obj_poi: Object points
        :param img_poi: Image Points
        :return: error, camera matrix, distortion coefficients
        """
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_poi, img_poi, (self.w, self.h), None,
                                                                           None)
        return rms, camera_matrix, dist_coefs

    def print_results(self):
        """Print Results"""
        print('\nRMS:\n', self.rms)
        print('\nCameramatrix:\n', self.camera_matrix)
        print("\ndistortion coefficients: ", self.dist_coefs.ravel())

    def save_parameters(self):
        """Save calculated parameters to results folder"""
        np.save(self.results_fldr + '/Cameramatrix.npy', self.camera_matrix)
        np.save(self.results_fldr + '/DistortionCoeffs.npy', self.dist_coefs)

    def prep_calib(self, inputfolder):
        """Prepare calibration
        :param inputfolder: folder of calibration images
        :return: Image Points, Object points

        """
        img_points = []
        obj_points = []
        files = glob.glob(inputfolder + '*.jpg')    # get all jpg-files in folder
        for fileidx, fn in enumerate(files):    # iterate over files
            gray_image = cv2.imread(fn, 0)    # read image as grayscale
            gray_image = cv2.equalizeHist(gray_image)
            if fileidx is 0:
                self.h, self.w = gray_image.shape[:2]    # get with and height
            # find the chesssboard corners in both images
            found, corners = cv2.findChessboardCorners(gray_image, self.pattern_size)
            # prepare camera calibration for this image if corners are found
            if found:
                cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), self.term)    # refine corner location
                img_points.append(corners.reshape(-1, 2))    # append corners
                obj_points.append(self.pattern_points)    # append pattern points
        return img_points, obj_points

    def print_chesssboard(self, img, corners, found, filename, save_folder):
        """Draws the found chesboard into the im age and saves it to a specified folder"""
        vis_l = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis_l, self.pattern_size, corners, found)
        path_l, name_l, ext_l = self.splitfn(filename)
        cv2.imwrite('%s/%s_chessboard.png' % (save_folder, name_l), vis_l)

    def check_for_folder(self, folder):
        """checks if folder exists, creates it if neccessary """
        if not os.path.exists(folder):
            os.makedirs(folder)

    def splitfn(self, fn):
        """Splits the filename in path, name and extension"""
        path, file = os.path.split(fn)
        name, ext = os.path.splitext(file)
        return path, name, ext


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calibrates a camera')
    parser.add_argument('folder_images', type=str, help='The folder containing the calibration images')
    parser.add_argument('results_folder', type=str, help='The folder where the results are saved')
    args = parser.parse_args()
    # assigns the command line argument (usually the video file) to video_src
    images_folder = args.folder_images
    results_folder = args.results_folder
    Calibration(images_folder, results_folder, save=True).run()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
