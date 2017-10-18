## Advanced Lane Finding Project

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[final_text]: ./output_images/final_text.png
[final]: ./output_images/final.png
[find_curve]: ./output_images/find_curve.png
[window_fitting]: ./output_images/window_fitting.png
[undistorted_warped]: ./output_images/undistorted_warped.png
[example_undist_warp]: ./output_images/example_undist_warp.png
[binary_pipeline]: ./output_images/binary_pipeline.png
[binary_threshold]: ./output_images/binary_threshold.png
[Undistorted]: ./output_images/Undistorted.png
[chessboard_undistort]: ./output_images/chessboard_undistort.png

[project_video]: ./output_videos/project_video.mp4

I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and also describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][chessboard_undistort]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][Undistorted]

#### 2. Using color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color, gradient, magnitude, HLS, and HSV thresholds to generate a binary image.  Here's an example of each method output for this step.

```python
gradx = abs_sobel_thresh(test_img, orient='x', thresh_min=70, thresh_max=255)
grady = abs_sobel_thresh(test_img, orient='y', thresh_min=50, thresh_max=255)
mag_binary = mag_thresh(test_img, sobel_kernel=3, mag_thresh=(90, 128))
dir_binary = dir_threshold(test_img, sobel_kernel=15, thresh=(0.9, 1.1))
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
hls_binary = hls_select(test_img, thresh=(90, 255))
hsv_binary = hsv_select(test_img, thresh=(90, 255))
```

![alt text][binary_threshold]

#### 3. Performing a perspective transform.

Demonstration of warping an image.
![alt text][example_undist_warp]

The code for my perspective transform appears in step 4 in the file `advanced_lane_lines.ipynb`. Using `cv2.getPerspectiveTransform` with source (`src`) and destination (`dst`) points to obtain perspective transform and inverse perspective transform matrix. Using `cv2.warpPerspective` to warp images. I chose to hardcode the source and destination points in the following value:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 445      | 320, 0        | 
| 240, 719      | 320, 719      |
| 1120, 719     | 960, 719      |
| 710, 445      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][undistorted_warped]

#### 4. Identified lane-line pixels and fit their positions with a polynomial

In step 5 and 6,

![alt text][window_fitting]

![alt text][find_curve]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In step 6, there is a `get_curvature(ploty, left_fit, right_fit, leftx, rightx, y_axis_left, y_axis_right)` and a `get_pos_center(left_fitx, right_fitx)` on the last section.

#### 6. Warp the detected lane boundaries back onto the original image such that the lane area is identified clearly.

I implemented this step in step 7 in my code in `advanced_lane_lines.ipynb` in the function `warp_lane_back()`. Using inverse perspective transform matrix to warp the polynomial lane line back to original lane line. Here is an example of my result on a test image:

![alt text][final_text]

---

### Pipeline (video)

#### 1. A link to my final video output.  My pipeline performs reasonably well on the entire project video. There are wobbly lines but no catastrophic failures that would cause the car to drive off the road!

Here's a [link to my video result][project_video]

---

### Discussion

#### 1. Create binary threshold images

The first problem is to create a decent threshold binary image, which would be used in later process. If my pipeline can not find lane lines correctly, the later process can not identify lane lines. I use RGB, HLS, and HSV binary method to create a threshold
binary image like below. `binary_output[binary >= 1] = 1` selects pixels whose value are greater than 1.

```python
def binary_pipeline(image):
    img = np.copy(image)
    
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=70, thresh_max=128)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=70, thresh_max=128)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(90, 128))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.9, 1.1))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    y_hls_binary = hls_select(img, [20,120,80], [38,200,255])
    y_hsv_binary = hsv_select(img, [20,60,60], [38,174,250])

    w_rgb_binary = rgb_select(img, [202,202,202], [255,255,255])
    w_hsv_binary = hsv_select(img, [20,0,200], [255,80,255])
   
    binary = y_hls_binary + y_hsv_binary + w_rgb_binary + w_hsv_binary
    
    binary_output =  np.zeros_like(binary)
    binary_output[binary >= 1] = 1
    
    return binary_output

```

#### 2. Using sliding window to search lane pixels

After tuning the binary pipeline, the next important step is how to detect Lane Pixels. I modified codes on step 5 below after finding udacity's original code can not find lane pixels correctly.  

```python
# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
offset = window_width/2

# Find the best left centroid by using past left center as a reference      
l_min_index = int(max(l_center+offset-margin,0))
l_max_index = int(min(l_center+offset+margin,warped.shape[1]))

if np.argmax(conv_signal[l_min_index:l_max_index]) == 0:
    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index+offset+offset/2
else:
    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
```

**The really important point** is that if the current window can not detect any lane, the next window has to start from the center point of current window. Otherwise, the next window would shift to more left or right comparing to current window.

#### 3. Assessing the curve's correctness by calculating their curvature

However, there still are some frames would fail catastrophically. Thus, I used curvature of two detected lanes to decide whether the result polynomial is a reasonable fit. First, the curvature has to be greater than a reasonable value. Second, the difference between the current curvature and last stored curvature has to be within a reasonable range.
