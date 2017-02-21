import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Sourcepoints:
    """Get the source points for warping an image to birds eye view"""
    #define 4 source points src = np.float32([[,],[,],[,],[,]])
    def __init__(self, image, w=1280, h=720):
        self.img = image

        self.source_points = []    # left-top, left_bottom, right_top, right_bottom

        self.dest_points = np.float32([[0.25*w, 0],[0.25*w, h],[0.75*w, 0],[0.75*w, h]])

    def get_transformation_matrix(self):
        """Calculates a transformation matrix and its inverse for image warping"""
        self.process(plots=False)

        M = cv2.getPerspectiveTransform(np.array(self.source_points).astype(np.float32), self.dest_points)
        M_inv = cv2.getPerspectiveTransform(self.dest_points, np.array(self.source_points).astype(np.float32))
        return M, M_inv

    def process(self, plots=True):
        """Find lane lines within the image
            param: self.img: source image
            param: plots: save the plotted results (T/F)

            return: image with drawn lane lines
        """
        # DEFINE RELEVANT PARAMETERS
        # orange RGB range
        orange_low = (200, 128, 0)
        orange_high = (255, 255, 150)
        # white RGB range
        white_low = (224, 224, 224)
        white_high = (255, 255, 255)

        # canny edge
        canny_low = 105
        canny_high = 210

        # dilate
        kernel = np.ones((3, 3), np.uint8)

        # hough lines parameters
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = 2 * np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 40  # minimum number of pixels making up a line
        max_line_gap = 80  # maximum gap in pixels between connectable line segments

        # line sorting, minimum slope
        min_slope = 0.45

        # START PIPELINE
        # get binary images from thresholding rgb-images with thresholds for...
        # ...orange lines
        orange = cv2.inRange(self.img, orange_low, orange_high)
        # ...white lines
        white = cv2.inRange(self.img, white_low, white_high)
        # combine the images, for a binary image containing white and orange pixels
        gray = cv2.bitwise_or(orange, white)
        # canny edge detection
        edges = self.canny(gray, canny_low, canny_high)
        # morphologic operations: dilate
        edges = cv2.dilate(edges, kernel, iterations=1)

        # apply mask to edges
        y_max, x_max = gray.shape  # img shape
        vertices = np.array([[(x_max * 0.05, y_max), (x_max * 0.47, 0.6 * y_max),
                              (x_max * 0.53, 0.6 * y_max), (x_max * 0.95, y_max)]], dtype=np.int32)
        masked_edges = self.region_of_interest(edges, vertices)

        # get hough lines from masked edges
        lines = self.get_hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

        # sort the lines in left and right lane line, get x-, y-values and length for each line
        left_x, left_y, right_x, right_y, left_len, right_len = self.sort_lines(lines, min_slope)

        # fit left and right line, draw them in a black image
        line_img = np.zeros_like(self.img)
        self.fit_lane_line(left_x, left_y, left_len, 1, line_img)
        self.fit_lane_line(right_x, right_y, right_len, 1, line_img)
        # get output image: combined lines and source image
        output_img = self.weighted_img(line_img, self.img, alpha=0.8, beta=1., gamma=0.)

        # plot images if plots is true
        if plots:
            # plot and save source img
            plt.figure(1)
            plt.title('Source Image with Lane Lines')
            plt.imshow(output_img)
            #plt.savefig(directory + 'Lane_Lines_' + file, bbox_inches='tight')

            # plot binary with white and yellow pixels
            plt.figure(2)
            plt.title('Binary for orange/white pixels')
            plt.imshow(gray, cmap='gray')

            # plot and save raw lines
            plt.figure(3)
            plt.title('Raw Hough Lines')
            raw_lines = self.hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
            plt.imshow(raw_lines)
            #plt.savefig(directory + 'Raw_Lines_' + file, bbox_inches='tight')

            # plot hough lines that pass the slope criterion
            plt.figure(4)
            hough_image = self.img.copy()
            plt.title('Hough Lines that pass the slope criterion')
            self.draw_lines(hough_image, lines, min_slope=0.3, color=[255, 0, 0], thickness=2)
            plt.imshow(hough_image)
            plt.show()


    def fit_lane_line(self, x_values, y_values, lengths, degree, image, min_y_factor=0.62):
        """Fit a lane line with np.polyfit and plot it in image
            param: x_values: x values of all HoughLines assigned to this lane line
            param: y_values: y values of all HoughLines assigned to this lane line
            param: degree: polynomial degree of the fit
            param: image: image to draw the line in
            param: min_y_value: the min y value of the drawn line (top point)

            return: image
        """
        if x_values and y_values:
            y_max = image.shape[0]  # max y value of the image
            min_y_value = min_y_factor * y_max  # min y value, i.e. y value of the top of the lane line
            # calc weights dependend of line length
            weights = np.divide(lengths, sum(lengths))
            line = np.poly1d(np.polyfit(y_values, x_values, degree, w=weights))  # construct polynomial
            point1 = (int(line(min_y_value)), int(min_y_value))  # 1st point of line (top)
            point2 = (int(line(y_max)), int(y_max))  # 2nd point of line (bottom)
            cv2.line(image, point1, point2, color=[255, 0, 0], thickness=10)  # draw line in image
        self.source_points.extend([list(point1), list(point2)])
        return image

    def sort_lines(self, lines, min_slope):

        """ sort lines in right and left lane line by checking for the minimum slope"""
        # get empty lists for results
        left_x, left_y, right_x, right_y, left_len, right_len = [], [], [], [], [], []
        # iterate over lines
        for line in lines:
            x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3],
            slope = 1.0 * (y2 - y1) / (x2 - x1)  # calc slope of line
            l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # calc length of line

            # filter by slope: right - positive above threshold min_slope; left - negative below threshold -1*min_slope
            if slope > min_slope:
                right_x.extend([x1, x2])  # right x values
                right_y.extend([y1, y2])  # right y values
                right_len.extend([l, l])  # line length
            elif slope < -1 * min_slope:
                left_x.extend([x1, x2])  # left x values
                left_y.extend([y1, y2])  # left y values
                left_len.extend([l, l])  # line length
        return left_x, left_y, right_x, right_y, left_len, right_len

    def get_hough_lines(self, image, rho, theta, threshold, min_line_len, max_line_gap):
        """find hough lines in image, return lines"""

        return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                               maxLineGap=max_line_gap)


    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, min_slope=0.3, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = 1.0 * (y2 - y1) / (x2 - x1)
                # filter by slope: right - positive above threshold min_slope; left - negative below threshold -1*min_slope
                if np.abs(slope) > min_slope:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    def weighted_img(self, img, initial_img, alpha=0.8, beta=1., gamma=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * alpha + img * beta + gamma
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """

        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

def main():
    #import argparse
    #parser = argparse.ArgumentParser(description='Calibrates a camera')
    #parser.add_argument('folder_images', type=str, help='The folder containing the calibration images')
    #parser.add_argument('results_folder', type=str, help='The folder where the results are saved')
    #args = parser.parse_args()
    ## assigns the command line argument (usually the video file) to video_src
    #images_folder = args.folder_images
    #results_folder = args.results_folder
    Sourcepoints().process()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
