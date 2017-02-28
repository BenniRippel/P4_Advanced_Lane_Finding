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
[image2]: ./output_images/test_warped.png "Warped Image"
[image3]: ./output_images/test_threshold.png "Binary Image"
[image4]: ./output_images/test_lane_detection.png "Fitted Lane"
[image5]: ./output_images/result.png "Result Image"

[video1]: ./project_video_output.mp4 "Video"

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is located in the file 'Calibration.py'. It may be used as a standalone tool, although 
for this project it is implemented as an additional module within the projects main file 'P4.py'



![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

###Pipeline (video)

####Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

![alt text][video1]
###Discussion

####Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?