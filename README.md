**Advanced Lane Finding Project**


In this project, the goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing 
camera on a car. The resulting pipeline should be robust to color change, e. g. shadows or pavement color.
 
The steps are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Compute the transformation matrix to perform perspective transform on a relevant part of the road  
* Apply a perspective transform to rectify an undistorted image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undistortion.png "Undistorted"
[image2]: ./output_images/test_undistortion2.png "Undistorted Test Image"
[image3]: ./output_images/test_warped.png "Warped Image"
[image4]: ./output_images/test_threshold.png "Binary Image"
[image5]: ./output_images/test_lane_detection.png "Fitted Lane"
[image6]: ./output_images/result.png "Result Image"

[video1]: ./project_video_output.mp4 "Video"

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in the file 'Calibration.py'. It may be used as a standalone tool, although 
for this project it is implemented as an additional module within the projects main file 'P4.py'.

The calibration matrix and the distortion coefficients are neccessary for undistorting images taken by a camera. All
camera images are distorted due to lens imperfections and manufacturing. As a result, straight lines are
displayed as curved lines in the image, with the degree of distortion depending on the location in the image. As we want
to calculate the real world radius of curvtature for the lane lines, an undistorted image is needed to reduce the error.

In the projects main file 'P4.py', the Calibration-Module is called within the function 'get_calibration_data' in line 480.  This function tries to load previously calculated calibration data from './calibration_parameters/'. If this fails,
the camera is calibrated with the images from './camera_cal/ and the results are stored as .npy-files. The function throws an error if both approaches fail.

The calibration itself is located within the class 'Calibration' in the file 'Calibration.py'. Apart from different
helper-functions like saving or printing the results, the class consists of a few core-functions:

* **run**
    This function runs the whole calibration process, containing preprocessing, calibration, saving and returning the
    desired data
* **prep_calib**
    Here, the preprocessing takes place. For every image within the specified input-folder, the histogram of intensity
    is equalized and potential chessboard corners of a given pattern are found using 'cv2.findChessboardCorners'. If the
    corners were found, the function 'cv2.cornerSubPix' refines the corner location for improved calibration accuracy.
    The found corners, which are pixel-coordinates on the image, are appended to the list 'img_points', while the
    pattern points, 3D Points within the chessboard coordinate system ( resulting in Z=0) are appended to the list
    'obj_points'.
* **calibrate**
    The above generated lists are passed to this function, where 'cv2.calibrateCamera' is called, calculating the
    camera_matrix, the distortion coefficients and the rms-error of the calibration process.
    
The success of the calibration can be tested using the function 'test_undistortion' in the file 'P4.py', which will 
result in the following image, showing a distorted calibration image and the same image with removed distortions. 

![alt text][image1]


Camera Matrix:

``
[[  1.16008925e+03   0.00000000e+00   6.64903615e+02]
``
``
 [0.00000000e+00   1.15663362e+03   3.88209795e+02]
``
``
 [0.00000000e+00   0.00000000e+00   1.00000000e+00]]
``


Distortion Coefficients:

``
[-0.23707184 -0.09496979 -0.00138167 -0.00028962  0.10926155]]
``

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate the distortion effects on images taken from the front of a car, the function 'cv2.undistort' was used 
with one of the test images. The distortionis not as obvious as in the above image, but definitely present.


![alt text][image2]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
- perspective transform used  with function 'cv2.warpPerspective', needs transformation matrix

- transformation matrix is obtained using the function 'get_transform_matrix' of the class 'TransformationMatrix' in the 
file 'TransformationMatrix.py', which is imported to 'P4.py' as an additional module. It is called within the function 
'get_transform_matrices', defined in lines 443-448 (file 'P4.py'). 

- the class 'TransformationMatrix' reuses the code of 'Project 1: Lane Lines', where lane lines were detected 
using simpler thresholding techniques and hough-lines. The fitted polynomials of an input image with straight lines are 
used to extract 4 source points, defining an isosceles trapezoid, lying on the lane lines.

source points
x | y
--|--
556| 474
222| 719 
725| 474 
1114| 719

- 

![alt text][image3]
####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 
![alt text][image4]



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

###Pipeline (video)

####Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

![alt text][video1]
###Discussion

####Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?