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

#### 1. Discussion of problems / issues I faced in my implementation of this project.  Where will my pipeline likely fail?  What did I do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The first problem is to create a decent threshold binary image, which would be used in later process. If my pipeline can not find lane lines correctly, the later process can not identify lane lines. I use combined, HLS, and HSV binary method to create a threshold
binary image like below. However, it didn't use all three method to create a binary image. It chose 2 out of 3 randomly by this line. `binary_output[binary >= 1] = 1`.

```python
def binary_pipeline(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(image)
    
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=70, thresh_max=128)
    grady = abs_sobel_thresh(img, orient='y', thresh_min=70, thresh_max=128)
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(90, 128))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.9, 1.1))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    hls_binary = hls_select(img, thresh=(85, 255))
    hsv_binary = hsv_select(img, thresh=(85, 255))

    binary = combined + hls_binary + hsv_binary
    #print(binary.shape)
    
    binary_output =  np.zeros_like(binary)
    binary_output[binary >= 1] = 1
    
    return binary_output= 0
    return binary
```

One thing to note is that I use L and S channel of HLS color space and S and V channel of HSV color space. Here is how it was implemented.

```python
def hsv_select(img, thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # H in [0,180], S and V in [0,255]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    bin_s = np.zeros_like(s_channel)
    bin_v = np.zeros_like(v_channel)
    bin_s[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    bin_v[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    binary_output = np.zeros_like(hsv[:,:,0])
    binary_output[(bin_s==1) & (bin_v==1)] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) # H in [0,180], L and S in [0,255]
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    bin_s = np.zeros_like(s_channel)
    bin_s[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    bin_l = np.zeros_like(l_channel)
    bin_l[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    
    binary_output = np.zeros_like(hls[:,:,0])
    binary_output[(bin_s==1) & (bin_l==1)] = 1
    return binary_output
```

After tuning the binary pipeline, there are still some frames whose lane lines can not be identify correctly. I think it's not efficient to stick to tune the binary pipeline.

In sliding window search, if there are many noises on the images, the noises would affect the searching process. Therefore, I use following lines to remove the noises that exist on the side of images.

```python
warp_bin_undist[:,0:warp_bin_undist.shape[1]//4-150] = 0
warp_bin_undist[:,warp_bin_undist.shape[1]*3//4+150:] = 0
warp_bin_undist[:,warp_bin_undist.shape[1]//2-100:warp_bin_undist.shape[1]//2+100] = 0
```

However, there still are some frames would fail catastrophically. Thus, I used curvature of two detected lanes to decide whether the result polynomial is a reasonable fit. First, the curvature has to be inside a reasonable range. Second, the difference between the current curvature and last stored curvature has to be small enough.
