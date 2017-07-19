# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale. 
Second, I convolute the image with a gaussian kernel in order to smooth the image.
Third, filtering the image with a canny filter with minimum threshold 50 and maximum threshold 150 to find out the edges of the image.
Fourth, keeping the area which we care about by croping the image with a intersection of quadrangle.
Fifth, using Hough transform to convert image space to polar space to find out the best lines from many candidate lines. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function.
First, dividing the lines into two group, left and right, by calculating the slopes of the lines.
Second, calculate the mean of slope of left lines.
Third, in the group of left lines, picking the top right-most point. According to straight line equation, y = ax + b. I can derive b give x, y, and a.
Fourth, using top right-most point, the known straight line and y equals 540 to find out x coordinate of the bottom left-most point.
Fifth, connect top right-most and bottom left-most point to get left line. Also, using step 1 to 4 to find out right line.


If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


1. I find out the filtering step is very important to the whole process. If the brightness of the video vary a lot, the output of canny will be very bad which could cause disaster to later process.
2. The quadrangle intersection I use is customised to a camera of the specfic fixing angle. If the camera is deflected or the lane width vary, the output of intersection will mess things a lot.



### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
