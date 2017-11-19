# Vehicle Detection

In this project, I will write a software pipeline to detect vehicles in a video.  

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vectors. 
* Normalize feature vectors and randomize a selection for training and testing.
* Train a SVM classifier.
* Implement a sliding-window technique and search for vehicles in images with a trained SVM classifier.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Important Note:   
*the value of png read by mpimg.imread() ranges from 0 to 1*  
*the value of jpg read by mpimg.imread() ranges from 0 to 255*  
*All codes are writen in vehicle_detection.ipynb*

[//]: # (Image References)
[car_not-car]: ./output_images/car_not-car.jpg
[spatial_hist]: ./output_images/spatial_hist.jpg
[car_hog]: ./output_images/car_hog.jpg
[window_size_32]: ./output_images/window_size_32.jpg
[window_size_48]: ./output_images/window_size_48.jpg
[window_size_64]: ./output_images/window_size_64.jpg
[window_size_96]: ./output_images/window_size_96.jpg
[window_overlap_0.65]: ./output_images/window_overlap_0.65.jpg
[window_overlap_0.75]: ./output_images/window_overlap_0.75.jpg
[window_overlap_0.85]: ./output_images/window_overlap_0.85.jpg
[slide_window]: ./output_images/slide_window.jpg
[test1]: ./output_images/test1.jpg
[test2]: ./output_images/test2.jpg
[test3]: ./output_images/test3.jpg
[test4]: ./output_images/test4.jpg
[test5]: ./output_images/test5.jpg
[test6]: ./output_images/test6.jpg
[series1]: ./output_images/series1.jpg
[series2]: ./output_images/series2.jpg
[series3]: ./output_images/series3.jpg
[series4]: ./output_images/series4.jpg
[series5]: ./output_images/series5.jpg
[series6]: ./output_images/series6.jpg
[sum_series]: ./output_images/sum_series.jpg
[box_heat]: ./output_images/box_heat.jpg

[out_project_video]: ./out_project_video.mp4

---

### Spatial Bins, Color Histogram, and Histogram of Oriented Gradients (HOG)

#### 1. Extracted Spatial Bins, Color Histogram, and HOG features from the training images. (1st code cell in vehicle_detection.ipynb)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes. (in code cell 2)  

![alt text][car_not-car]

I then extracted spatially binned features and color histogram. (in code cell 3)  

![alt text][spatial_hist]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:  (in code cell 4)  

![alt text][car_hog]

#### 2. Finding the best combination from Spatially Binned, Color Histogram and HOG parameters.

Picking `color_space = ['HSV', 'YUV', 'YCrCb']`, `orient = 9`, `pix_per_cell = [8,16]`, `cell_per_block = 2`, and turning on/off Spatially Binned, Color Histogram, there are totally `3x2x2 = 12` combinations. Here are the results. (in code cell 6)  

| color_space | pix_per_cell | Spatially Binned and Color Histogram | Testing Accuracy |
|:-----:|:-----:|:-----:|:-----:|
| HSV | 8 | ON | 0.9887 |
| HSV | 8 | OFF | 0.9817 |
| HSV | 16 | ON | 0.987 |
| HSV | 16 | OFF | 0.9662 |
| YUV | 8 | ON | 0.9904 |
| YUV | 8 | OFF | 0.9809 |
| YUV | 16 | ON | 0.9913 |
| YUV | 16 | OFF | 0.971 |
| YCrCb | 8 | ON | 0.9879 |
| YCrCb | 8 | OFF | 0.9794 |
| YCrCb | 16 | ON | 0.9904 |
| YCrCb | 16 | OFF | 0.9721 |

Therefore, I picked `color_space = YUV`, `orient = 9`, `pix_per_cell = 16`, `cell_per_block = 2`, Spatial Bins and Color Histogram as my SVM's training parameter.  

### Sliding Window Search

#### 1. Sliding window search. Determine window size and how much to overlap windows.

Using `window_size = 32, 48, 64, 96` to test on examle images helps to find out the final value used on pipeline. (in code cell 10)  
As you can see, `window_size = 32` is too small, which captue many false positive (FP) images. `window_size = 64` doesn't make much difference compare with `window_size = 96`, but 96 can capture more precisely when can is near to camera.
Therefore, I finally chose `window_size = 48, 96` to be the window sizes.  

![alt text][window_size_32]
![alt text][window_size_48]
![alt text][window_size_64]
![alt text][window_size_96]

Here are the result of using `window_overlap = 0.65, 0.75, 0.85` test on the example image respectively.  
As you can see, it doesn't make much difference with different overlap values. Therefore, I used average value 0.75 as the final step ratio. (in code cell 11)  

![alt text][window_overlap_0.65]
![alt text][window_overlap_0.75]
![alt text][window_overlap_0.85]

Showing all the sliding windows on an image. The range of y-axis is from 400 to 700. (in code cell 12)  

![alt text][slide_window]

#### 2. Show some examples of test images to demonstrate how my slide_window() and search_windows() is working. No further optimization methods provided in this step.

Ultimately I searched on two window scales and using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images: (in code cell 13)  

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]

---

### Video Implementation

#### 1. Here is a link to my final video output. My pipeline could perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes may occur during the driving).
Here's a [link to my video result](./out_project_video.mp4)

#### 2. Using heatmap to filter out false positives.

This block of code filters out some false positive boxes and combines overlaped boxes into one larger box. (in code cell 15)

```python
heat = np.zeros_like(image[:,:,0]).astype(np.float)
box_list = hot_windows
# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,heat_threshold)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
heat_window_img = draw_labeled_bboxes(cp_image, labels, (0,0,255))
```

Before applying heatmap:  

![alt text][series1]

After applying heatmap:  

![alt text][box_heat]

---

### Discussion

#### 1. Brief discussions of problems / issues I faced in this project and how I solved it.

So far, all the methods used in this project are pretty standard technics, but there still exist two problems.  

First, false positive boxes were still captured by the SVM classifier. Adding a queue `q_fp = deque(maxlen=4)` with following codes could solve this problem.  

```python
    q_fp.append(heat)
    heat = apply_threshold(sum(q_fp)/float(len(q_fp)), 4)
```

In the previous heatmap process, all frames are considered respectively. The summatioin of boxes are used by later steps only in current frame. The boxes have no connections from the previous frame to the current frame. But now, using `q_fp` to store the summation of boxes in a row, it can filter out false positive boxes. The feasibility of this method is based on the fact that the false positive boxes would not occur continuously, at least less than 4. If the FP boxes occur continuously, this method may not work.  

The second problem is how to smooth out the wobbly boxes. Again, similar to previous method, using a queue, `q_wobbly = deque(maxlen=12)`, can smooth out the wobbly boxes. The more elements `q_wobbly` has, the smoother the final boxes. But the boxes may have delay when marking the vehicles.  
```python
    q_wobbly.append(heatmap)
    labels = label(sum(q_wobbly))
```