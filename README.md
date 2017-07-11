**Advanced Lane Finding Project**


In this project, the goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing 
camera on a car. The resulting pipeline should be robust to color change, e. g. shadows or pavement color.
 
The steps are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Compute the transformation matrix to perform perspective transform on a relevant part of the road  
* Apply a perspective transform to rectify an undistorted image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undistortion.png "Undistorted"
[image2]: ./output_images/test_undistortion2.png "Undistorted Test Image"
[image3]: ./output_images/test_warped.png "Warped Image"
[image4]: ./output_images/test_threshold.png "Binary Image"
[image5]: ./formula.png "Formula curvature"
[image6]: ./output_images/test_lane_detection.png "Fitted Lane"
[image7]: ./output_images/result.png "Result Image"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in the file 'Calibration.py'. It may be used as a standalone tool, although 
for this project it is implemented as an additional module within the projects main file 'P4.py'.

The calibration matrix and the distortion coefficients are neccessary for undistorting images taken by a camera. All
camera images are distorted due to lens imperfections and manufacturing. As a result, straight lines are
displayed as curved lines in the image, with the degree of distortion depending on the location in the image. As we want
to calculate the real world radius of curvtature for the lane lines, an undistorted image is needed to reduce the error.

In the projects main file 'P4.py', the Calibration-Module is called within the function 'get_calibration_data' in line 502.  
This function tries to load previously calculated calibration data from './calibration_parameters/'. If this fails,
the camera is calibrated with the images from './camera_cal/ and the results are stored as .npy-files. The function 
throws an error if both approaches fail.

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


**Camera Matrix:**

| . | 1st col |  2nd col | 3rd col  | 
|:---|:---:|:---:|:---:|
|**1st row**|1.16008925e+03  | 0.00000000e+00 |  6.64903615e+02|
|**2nd row**|0.0000000e+00  | 1.15663362e+03  | 3.88209795e+02|
| **3rd row**|0.00000000e+00  | 0.00000000e+00  | 1.00000000e+00|


**Distortion Coefficients:**

| k1 | k2| p1| p1| k3|
|---|---|---|---|---|
|-0.23707184 | -0.09496979| -0.00138167| -0.00028962|  0.10926155|


### Pipeline (single images)

The code for finding lane lines on a single image is located in the file 'P4.py'. Within the class 'P4' a function 
'run_image' is defined (line 52), that reads a selected image and undistorts it. The undistorted image is converted into 
a binary image using color and gradient thresholding. Then, the binary image gets transformed to a bird's eye perspective.
The function 'fit_lines' is used with this warped binary image, and calculates the polynomial representation of both lane lines,
as well as the radius of curvature of the lane and the offset of the car to the lane center. The function 'draw_warped' 
draws the lane lines in the original image. All these steps are described below.


#### 1. Provide an example of a distortion-corrected image.
To demonstrate the distortion effects on images taken from the front of a car, the function 'cv2.undistort' was used 
with one of the test images. The effect is not as obvious as in the above image, but definitely present.


![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
A binary image is created in the function 'threshold_image' (line 346, P4.py), where the binary images created by 
gradient thresholds on the S-colorchannel and the L-colorchannel of an HLS-Image are combined with binary images created 
by color thresholding the S- and L-channel. All 4 binary images are combined using 'OR'- conditions.

**Gradient thresholding** is defined in the function 'grad_thresh' (line 373), where 'Magnitude of the Gradient'
(function 'mag_thresh', line 416) and 'Direction of the Gradient' (function 'dir_thresh', line 429) thresholds are used.
An additional color threshold is used, to exclude dark image regions (tire markings etc.) from the binary images. All 3 
binary images are combined using the 'AND'-condition. 

The thresholds used are:
L-Channel:

| . | Low Thresh | High Thresh|
|---|---|---|
|**Mag**|20 | 255|
|**Dir**|-0.56 | 1.3|
|**L-Chan**|160 | 255|

S-Channel:

| . | Low Thresh | High Thresh|
|---|---|---|
|**Mag**|20 | 120|
|**Dir**|-0.56 | 1.3|
|**L-Chan**|120 | 255|
  
  
**Color thresholding** is defined in the function 'combine_color' (line 405), where the L- and S-channel of an HLS-image are 
thresholded using the function 'select_color' (line 399) and combined with an 'OR'-condition.

The next image shows an example of a binary image.
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
A perspective transform is performed, using the function 'cv2.warpPerspective', which needs a transformation matrix to work.

The transformation matrix is obtained using the function 'get_transform_matrix' of the class 'TransformationMatrix' in the 
file 'TransformationMatrix.py', which is imported to 'P4.py' as an additional module. It is called within the function 
'get_transform_matrices', defined in lines 466-471 (file 'P4.py'). 

The class 'TransformationMatrix' reuses the code of 'Project 1: Lane Lines', where lane lines were detected 
using simpler thresholding techniques and hough-lines. The fitted polynomials of an input image with straight lines are 
used to extract 4 source points lying on the lane lines, defining an isosceles trapezoid. 

**Source Points:**

| x | y |
|:---:|:---:|
|556 | 474 |
|222 | 719 |
|725 | 474 |
|1114 | 719 |

These source points are mapped to the following destination points, with w being the image width and h the image height

**Destination Points:**

| x | y |
|:---:|:---:|
|0.2*w | 0 |
|0.2*w | h |
|0.8*w | 0 |
|0.8*w | h |

Using these points with the function 'cv2.getPerspectiveTransform' computes the transformation matrix and its inverse matrix.
The transformation matrix is used for all perspective transformations in this project, assuming that the camera is not 
moving relative to the ground. This assumption is true for the most parts of the project video.

Once the transformation matrix is calculated, the perspective transformation can take place. This procedure is 
located in the function 'warp_image' (line 441, P4.py), using the function 'cv2.warpPerspective'. When processing an image, 
warping happens right after color and gradient thresholding (line 57 in function 'run_image' and line 67 in function 
'process frame', file 'P4.py').

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
After preprocessing the input image resulting in a perspective transformed binary image, the lane lines are identified 
within the function 'fit_lines' (lines 176-196). 

##### Single Image 
For the **single image pipeline**, the function 'find_new_lines' is called (defined in lines 216-308, called in line 182).
This function represents the sliding window approach presented in the Udacity lectures. To find the approximate x-location of the 
lane lines in the bottom half of the image, a histogramm of all pixels withing the warped binary image is calculated. It 
is assumed that the two highest peaks of the histogram resemble the lane lines. To find the valid pixels of each line 
within the binary image, a window with 1/9th times the image height and a width of 200px is centered around the previously 
found x-location of the lines. All pixels within this window are considered valid. The x-location for the next window, 
which is fitted above the first one, is calculated by the median x-location of the previous window. These steps are 
repeated until the full image height is fitted with windows (9 windows in this case). The found valid pixels are used to 
fit a second order polynomial for each line. Additionally, the polynomials are also fitted in real world space, by 
multiplying the pixel coordinates with conversion parameters.
 
The image below shows the windows and valid pixels found in this process.

![alt text][image5]

##### Video
The function 'fit_lines' has a different behaviour when a video (several successive images) are processed. In this case,
temporal information of the lines is saved to filter line fits and improve lane detection with knowledge about previous 
lane fits.

The function 'fit_lanes' now checks for valid line fits found in the last 5 frames. If there are no valid fits, the 
function 'find_new_lines' is called with the above functionality. If valid lines exist, the function 'find_line_by_margin'
 is called (defined in lines 310-344, called in 185). New valid pixels are found in an area with a width of 200px around 
 the mean valid line fits, and new polynomials are fitted to those valid pixels.
 
If the new lane fits pass the lane distance check (defined: lines 198-205, called: line 188), which tests that the 
distance between the lanes is in the range of 2m to 6m, they are appended to the list of valid fits.
  
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The radius of curvature and the position of the vehicle with respect to the lane center are calculated within the function 
'calc_radius_and_offset', defined in lines 108-126. This function gets called within the function 'fit_lines' (line 193),
but only if the current fit is valid.

For the calculation of the radius, the coefficients of the lane line fits in real world space are used. The left and 
right radius are calculated by the following formula

![alt text][image5]

The offset of the car to the lane center is calculated by subtracting 0.5*ImageWidth from the mean pixel position of 
both lanes at the nearest distance, and multiplying this value by 3.7/700, which is a scaling factor to project horizontal 
pixel distances in meters.
 
The mean value of both calculated values, radius and offset, are printed in every output image by the function 
'write to frame' (lines 99-106) 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
![alt text][image6]
### Pipeline (video)
As described above, the main difference between the single image pipeline and the video pipeline is the temporal information
shared between frames. This helps to smooth the output. This is noticeable when the car drives over bumps or through shadows.

#### Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)
The processed project video can be found here:

./project_video_output.mp4

The challenge video is:

./challenge_video_output
### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
**Problems during implementation**

The hardest part was tuning the gradient and color thresholds, especially for the challenge video. Tuning those parameters 
by hand and experimenting with all available options takes a lot of time.Consequently,  I built a tool in 
'tuneThreshold.py' that uses trackbars to tune the parameters and switch between test images. For the 
challenge video, I created additional test images. The tool allows tuning all parameters for all test images in a few minutes.

**Where will the pipeline fail**

 - small radius of curvature:
    The lane lines will not fit on the warped image entirely, leaving the image on the side. Thus a lot of valid pixels 
    will be lost decreasing the quality of the fit
 - S -curves
    The 'fit line'-function only fits a second degree polynomial that cant handle S shapes
 - dirty roads, difficult light etc
    All cases when the color thresholds will fail, the pipeline will have a hard time, relying only on gradient thresholds
    Catching those difficult cases takes a lot of additional code (and might still not work)
 - steep slopes, heavy loads, heavy vehicle movement
    A static transformation matrix will not work anymore

**Possible Improvements**

Some problems may be solved by widening the area the source points cover, so the warped image includes more of the area 
to the left/right of the lane. This will help with losing valid pixels due to sharp turns. Fitting higher degree polynomials might 
complicate the whole process due to wild oscillation, but will help detecting S-shaped curves if fitting works. Different
  thresholds for gradient and color thresholding for different environment conditions might be a good idea, too.
