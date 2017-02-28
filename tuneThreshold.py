import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

class TuneThreshold:

    def __init__(self, file='./test_images/test5.jpg'):
        self.img = cv2.imread(file)
        self.prep_img = []

        # L channel:
        # Kombo aus dir und mag mit mag (30,255,5) und dir (101, 287,7)
            # im schatten super, auf hellem boden kacke für gelbe linien
        # Kombo aus x und y mit x(30, 255,7) und y(30, 255,7)

        # S channel:
        # Kombo aus dir und mag mit mag (20,120+255,5) und dir (101, 287,7)
            # ohne schatten gut, im schatten nich so (da muss min runter (10-15), vorsicht wegen artefakten)
        # Kombo aus x und y mit x(30, 255,7) und y(30, 255,7)


    def combine_grad(self):
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 1]

        th_x=self.abs_sobel_thresh(orient='x', sobel_kernel=7, lower=30, upper=150)
        th_y=self.abs_sobel_thresh(orient='y', sobel_kernel=7, lower=35, upper=150)

        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 1]
        th_m=self.mag_thresh(sobel_kernel=5, lower=35, upper=120)
        th_d=self.dir_threshold(sobel_kernel=7, lower=-0.56, upper=1.3)

        # Combine thresholds
        combined = np.zeros_like(self.prep_img)
        # combined[((th_m == 1) & (th_d == 1)) | ((th_x == 1) & (th_y == 1))] = 1
        combined[((th_m == 1) & (th_d == 1)) ] = 1

        c = np.zeros_like(self.prep_img)
        c[(combined == 1) | (self.combine_color()==1)] = 1


        fig=plt.figure()
        plt.title('Combined gradients')
        plt.imshow(combined, cmap='gray')

        fig = plt.figure()
        plt.imshow(c, cmap='gray')


        plt.show()
        # # preprocess image
        # self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 2]
        #
        # th_x=self.abs_sobel_thresh(orient='x', sobel_kernel=7, thresh=(30, 150))
        # th_y=self.abs_sobel_thresh(orient='y', sobel_kernel=7, thresh=(35, 150))
        # th_m=self.mag_thresh(sobel_kernel=7, thresh=(50, 150))
        # th_d=self.dir_threshold(sobel_kernel=15, thresh=(0.8, 1.2))
        #
        # # Combine thresholds
        # comb = np.zeros_like(self.prep_img)
        # comb[((th_m == 1) & (th_d == 1)) | ((th_x == 1) & (th_y == 1))] = 1
        #
        # fig=plt.figure()
        # plt.imshow(comb, cmap='gray')
        #
        # fig = plt.figure()
        # c = np.zeros_like(self.prep_img)
        # c[(combined == 1) | (comb==1) | (self.combine_color()==1)] = 1
        # plt.imshow(c, cmap='gray')
        #
        # plt.show()

    def grad_thresh(self, image, chan=1):
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,chan]
        # Apply each of the thresholding functions
        th_x=self.abs_sobel_thresh(channel, orient='x', sobel_kernel=7, thresh=(30, 150))
        th_y=self.abs_sobel_thresh(channel, orient='y', sobel_kernel=7, thresh=(35, 150))
        th_m=self.mag_thresh(channel, sobel_kernel=5, thresh=(35, 120))
        th_d=self.dir_threshold(channel, sobel_kernel=7, thresh=(-0.56, 1.3))
        ret = np.zeros_like(channel)
        ret[((th_m == 1) & (th_d == 1)) | ((th_x == 1) & (th_y == 1)) | (self.combine_color(image) == 1)] = 1

        return ret

    def combine_color(self):
        h = self.select_color(19, 25, 0)
        l = self.select_color(210, 246, 1)
        s = self.select_color(150, 180, 2)

        # Combine thresholds
        combined = np.zeros_like(h)
        combined[((h == 1) | (l == 1) | (s == 1))] = 1

        return combined

    def tune_gradient_on_thresh_color(self):
        # named window
        cv2.namedWindow('Tune')
        # test images
        images = glob.glob('./test_images/*.jpg')
        # create trackbars
        cb = self.null
        cv2.createTrackbar('Image', 'Tune', 0, len(images)-1, cb)
        cv2.createTrackbar('Kernel Dir', 'Tune', 1, 15, self.null)
        cv2.createTrackbar('Lower Threshold Dir', 'Tune', 0, 314, self.null)
        cv2.createTrackbar('Upper Threshold Dir', 'Tune', 0, 314, self.null)

        img_idx=0
        kernel_d=7
        lower_d=-0.56
        upper_d=1.3

        while True:
            self.img=cv2.imread(images[img_idx])
            # h = self.select_color(19, 25, 0)
            l = self.select_color(249, 255, 1)
            s = self.select_color(210, 255, 2)
            # Combine thresholds
            self.prep_img = self.img
            self.prep_img[((l == 0) | (s == 0))] = 0

            thresh = self.dir_threshold(kernel_d, lower_d, upper_d)
            comb = np.zeros_like(thresh)
            comb[(thresh == 1)] = 1
            cv2.imshow('gae', self.prep_img)
            cv2.imshow('Tune', 255*thresh)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            img_idx = cv2.getTrackbarPos('Image', 'Tune')
            kernel_d = cv2.getTrackbarPos('Kernel Dir', 'Tune')*2+1
            lower_d = (cv2.getTrackbarPos('Lower Threshold Dir', 'Tune')-157) /100.0
            upper_d = (cv2.getTrackbarPos('Upper Threshold Dir', 'Tune')-157) /100.0



    def show_color_thresholds(self):
        # named window
        cv2.namedWindow('Tune')
        # test images
        images = glob.glob('./test_images/*.jpg')
        # create trackbars
        cb = self.null
        cv2.createTrackbar('Image', 'Tune', 0, len(images)-1, cb)
        img_idx=0

        while True:
            self.img=cv2.imread(images[img_idx])
            # h = self.select_color(19, 25, 0)
            l = self.select_color(205, 245, 1)
            s = self.select_color(100, 180, 2)

            # Combine thresholds
            combined = np.zeros_like(s)
            combined[((l == 1) | (s == 1))] = 1

            cv2.imshow('Tune', 255*combined)
            cv2.imshow('IMG', self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            img_idx = cv2.getTrackbarPos('Image', 'Tune')

    def tune_color(self):
        # named window
        cv2.namedWindow('Tune')
        # test images
        images = glob.glob('./test_images/*.jpg')


        # create trackbars
        cb = self.select_color
        cv2.createTrackbar('Lower Threshold', 'Tune', 0, 255, cb)
        cv2.createTrackbar('Upper Threshold', 'Tune', 0, 255, cb)
        cv2.createTrackbar('Channel', 'Tune', 0, 2, cb)
        cv2.createTrackbar('Image', 'Tune', 0, len(images)-1, cb)
        lower = 0
        upper=255
        img_idx = 0
        chan=0

        while True:
            self.img=cv2.imread(images[img_idx])
            thresh = cb(lower, upper, chan)

            cv2.imshow('Tune', 255*thresh)
            cv2.imshow('Image', self.img)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            lower = cv2.getTrackbarPos('Lower Threshold', 'Tune')
            upper = cv2.getTrackbarPos('Upper Threshold', 'Tune')
            chan = cv2.getTrackbarPos('Channel', 'Tune')
            img_idx = cv2.getTrackbarPos('Image', 'Tune')

    def select_color(self, lower, upper, channel=2):
        chan = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, channel]
        ret = np.zeros_like(chan)
        ret[(chan<upper) & (chan>lower)] =1
        return ret

    def tune(self):
        # named window
        cv2.namedWindow('Tune')
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 1]
        # create trackbars
        cv2.createTrackbar('Lower Threshold X', 'Tune', 0, 255, self.null)
        cv2.createTrackbar('Upper Threshold X', 'Tune', 0, 255, self.null)
        cv2.createTrackbar('Lower Threshold Y', 'Tune', 0, 255, self.null)
        cv2.createTrackbar('Upper Threshold Y', 'Tune', 0, 255, self.null)
        lower_x, lower_y = 0, 0
        upper_x, upper_y=255, 255
        while True:
            thresh_x = self.abs_sobel_thresh(orient='x', sobel_kernel=7, lower=lower_x, upper=upper_x)
            thresh_y = self.abs_sobel_thresh(orient='y', sobel_kernel=7, lower=lower_y, upper=upper_y)

            thresh = np.zeros_like(thresh_x)
            thresh[(thresh_x==1) & (thresh_y==1)]=1

            cv2.imshow('Tune', 255*thresh)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            lower_x = cv2.getTrackbarPos('Lower Threshold X', 'Tune')
            upper_x = cv2.getTrackbarPos('Upper Threshold X', 'Tune')
            lower_y = cv2.getTrackbarPos('Lower Threshold Y', 'Tune')
            upper_y = cv2.getTrackbarPos('Upper Threshold Y', 'Tune')


    def tuneDir(self):
        # named window
        cv2.namedWindow('Tune')
        # preprocess image
        self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 1]
        # create trackbars
        #cb = self.dir_threshold
        cv2.createTrackbar('Kernel Dir', 'Tune', 1, 15, self.null)
        cv2.createTrackbar('Lower Threshold Dir', 'Tune', 0, 314, self.null)
        cv2.createTrackbar('Upper Threshold Dir', 'Tune', 0, 314, self.null)
        cv2.createTrackbar('Kernel Mag', 'Tune', 1, 15, self.null)
        cv2.createTrackbar('Lower Threshold Mag', 'Tune', 0, 255, self.null)
        cv2.createTrackbar('Upper Threshold Mag', 'Tune', 0, 255, self.null)


        kernel_d=7
        lower_d=-0.56
        upper_d=1.3
        kernel_m=5
        lower_m=34
        upper_m=121
        while True:
            self.prep_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:, :, 1]
            mag = self.mag_thresh(kernel_m, lower_m, upper_m)
            thresh = self.dir_threshold(kernel_d, lower_d, upper_d)
            # Combine thresholds
            combined = np.zeros_like(mag)
            combined[((mag == 1) & (thresh == 1))] = 1

            cv2.imshow('Tune', 255*combined)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            kernel_d = cv2.getTrackbarPos('Kernel Dir', 'Tune')*2+1
            lower_d = (cv2.getTrackbarPos('Lower Threshold Dir', 'Tune')-157) /100.0
            upper_d = (cv2.getTrackbarPos('Upper Threshold Dir', 'Tune')-157) /100.0
            kernel_m = cv2.getTrackbarPos('Kernel Mag', 'Tune')*2+1
            lower_m = cv2.getTrackbarPos('Lower Threshold Mag', 'Tune')
            upper_m = cv2.getTrackbarPos('Upper Threshold Mag', 'Tune')

    def null(self, *args):
        pass

    def abs_sobel_thresh(self, orient='x', sobel_kernel=3, lower=0, upper=255):
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
        binary_output[(scaled >= lower) & (scaled <= upper)] = 1
        return binary_output

    def mag_thresh(self, sobel_kernel=3, lower=0, upper= 255):
        """Apply 'Magnitude of the Gradient' threshold to an image channel """
        sob_x = cv2.Sobel(self.prep_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sob_y = cv2.Sobel(self.prep_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        mag = (sob_x ** 2 + sob_y ** 2) ** 0.5
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * mag / np.max(mag))
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled)
        binary_output[(scaled < upper) & (scaled > lower)] = 1
        return binary_output

    def dir_threshold(self, sobel_kernel=3, lower=0, upper= np.pi/2):
        """Apply 'Direction of the Gradient' threshold to an image channel """
        # Take the absolute value of the x and y gradients
        abs_x = np.absolute(cv2.Sobel(self.prep_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        abs_y = np.absolute(cv2.Sobel(self.prep_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        di = np.arctan2(abs_y, abs_x)
        # Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(abs_x)
        binary_output[(di > lower) & (di < upper)] = 1
        return binary_output

def main():
    TuneThreshold().show_color_thresholds()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
