# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidWhiteCurve]: ./test_images/solidWhiteCurve.jpg "solidWhiteCurve"
[solidWhiteCurve_gray]: ./test_images_output/solidWhiteCurve_gray.jpg "solidWhiteCurve_gray"
[solidWhiteCurve_blur_gray]: ./test_images_output/solidWhiteCurve_blur_gray.jpg "solidWhiteCurve_blur_gray"
[solidWhiteCurve_edges]: ./test_images_output/solidWhiteCurve_edges.jpg "solidWhiteCurve_edges"
[solidWhiteCurve_masked_edges]: ./test_images_output/solidWhiteCurve_masked_edges.jpg "solidWhiteCurve_masked_edges"
[solidWhiteCurve_output]: ./test_images_output/solidWhiteCurve.jpg "solidWhiteCurve_output"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
Original Image<br/>
![alt text][solidWhiteCurve]

My pipeline consisted of 5 steps.<br/>
First, I converted the images to grayscale.<br/>
![alt text][solidWhiteCurve_gray]

Second, I convoluted the image with a gaussian kernel in order to smooth the image.<br/>
![alt text][solidWhiteCurve_blur_gray]

Third, filtering the image with a canny filter with minimum threshold 50 and maximum threshold 150 to find out the edges of the image.<br/>
![alt text][solidWhiteCurve_edges]

Fourth, keeping the area which we care about by croping the image with a intersection of quadrangle.<br/>
![alt text][solidWhiteCurve_masked_edges]

Fifth, using Hough transform to convert image space to polar space to find out the best lines from many candidate lines.<br/>
![alt text][solidWhiteCurve_output]

How I modify draw_lines()<br/>
Step 1: Dividing the lines into two group, left and right, by calculating the slopes of the lines.<br/>
Step 2: Calculate the mean of slope of left lines.<br/>
Step 3: In the group of left lines, picking the top right-most point. According to straight line equation, y = ax + b. I can derive b give x, y, and a.<br/>
Step 4: Using top right-most point, the known straight line and y equals 540 to find out x coordinate of the bottom left-most point.<br/>
Step 5: Connect top right-most and bottom left-most point to get left line. Also, repeating step 1 to 4 to find out right line.

### 2. Identify potential shortcomings with your current pipeline
The first thing I found which is very important to the whole process is filtering the original images. If the brightness of the video varies a lot, the output image of canny filter will be very bad which could cause disaster to later process.<br/>
Second, the quadrangle intersection I use is customised to a camera of the specfic fixing angle. If the camera is deflected or the lane width vary, the output of intersecting process will also mess up the whole process.<br/>

### 3. Suggest possible improvements to your pipeline
To solve the first issue, subtracting the mean value from a original image may help. It can help to normalize every image which can make it easier to find out the correct parameter of filter.<br/>
I still don't know how to fix the second isses.<br/>

