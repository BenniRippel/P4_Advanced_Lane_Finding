import numpy as np
import cv2
import matplotlib.pyplot as plt

class TuneThreshold:

    def __init__(self, file='./test_images/test1.jpg'):
        self.img = cv2.imread(file)
        self.prep_img = []

        # Thresholds: good values for l channel (im l chan ist die gelbe linie fast nie da)
        # X Sobel: 30, 150, 7
        # Y Sobel: 35, 150, 7
        # Mag: 50, 150, 7
        # Dir: 80, 120, 15

        #TODO: statt l den s channel nehmen

    def combine(self):
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:,:,2]

        th_x=self.abs_sobel_thresh(orient='x', sobel_kernel=7, thresh=(30, 150))
        th_y=self.abs_sobel_thresh(orient='y', sobel_kernel=7, thresh=(35, 150))
        th_m=self.mag_thresh(sobel_kernel=7, thresh=(50, 150))
        th_d=self.dir_threshold(sobel_kernel=15, thresh=(0.8, 1.2))

        # Combine thresholds
        combined = np.zeros_like(self.prep_img)
        combined[((th_m == 1) & (th_d == 1))] = 1

        fig=plt.figure()
        plt.imshow(self.prep_img, cmap='gray')
        plt.show()

    def tune(self):
        # named window
        cv2.namedWindow('Tune')
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:,:,1]
        # create trackbars
        cb = self.mag_thresh
        cv2.createTrackbar('Lower Threshold', 'Tune', 0, 255, cb)
        cv2.createTrackbar('Upper Threshold', 'Tune', 0, 255, cb)

        lower = 0
        upper=255
        while True:
            thresh = self.mag_thresh(sobel_kernel=7, thresh=(lower, upper))
            cv2.imshow('Tune', 255*thresh)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            lower = cv2.getTrackbarPos('Lower Threshold', 'Tune')
            upper = cv2.getTrackbarPos('Upper Threshold', 'Tune')

    def tuneDir(self):
        # named window
        cv2.namedWindow('Tune')
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:,:,1]
        # create trackbars
        cb = self.dir_threshold
        cv2.createTrackbar('Lower Threshold', 'Tune', 0, 314, cb)
        cv2.createTrackbar('Upper Threshold', 'Tune', 0, 314, cb)

        lower = 0
        upper=np.pi/4
        while True:
            thresh = self.dir_threshold(sobel_kernel=15, thresh=(lower, upper))
            cv2.imshow('Tune', 255*thresh)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            lower = cv2.getTrackbarPos('Lower Threshold', 'Tune') /100.0
            upper = cv2.getTrackbarPos('Upper Threshold', 'Tune') /100.0


    def abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Apply 'Sobel Gradient' threshold to an image channel """
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(self.prep_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(self.prep_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the derivative or gradient
        abs = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled = np.uint8(255 * abs / np.max(abs))

        binary_output = np.zeros_like(scaled)
        binary_output[(scaled >= min(thresh)) & (scaled <= max(thresh))] = 1
        return binary_output

    def mag_thresh(self, sobel_kernel=3, thresh=(0, 255)):
        """Apply 'Magnitude of the Gradient' threshold to an image channel """
        sob_x = cv2.Sobel(self.prep_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sob_y = cv2.Sobel(self.prep_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        mag = (sob_x ** 2 + sob_y ** 2) ** 0.5
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * mag / np.max(mag))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled < max(thresh)) & (scaled > min(thresh))] = 1
        return binary_output

    def dir_threshold(self, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """Apply 'Direction of the Gradient' threshold to an image channel """
        # Take the absolute value of the x and y gradients
        abs_x = np.absolute(cv2.Sobel(self.prep_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_y = np.absolute(cv2.Sobel(self.prep_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        di = np.arctan2(abs_y, abs_x)
        # Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(abs_x)
        binary_output[(di > min(thresh)) & (di < max(thresh))] = 1
        return binary_output

def main():
    TuneThreshold().combine()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
