Behavioral Cloning Project
===========================

[//]: # (Image References)

[cover]: ./photo/cover.png "Cover"
[model_architecture]: ./photo/model_architecture.png "Model Architecture"
[center_flip]: ./photo/center_flip.png "Center Flip"
[center]: ./photo/center.png "Center"
[crop]: ./photo/crop.png "Crop"
[left]: ./photo/left.png "Left"
[right]: ./photo/right.png "Right"
[resize]: ./photo/resize.png "Resize"
[steering_hist]: ./photo/steering_hist.png "Steering Histogram"

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report
- Fulfill all [Rubric Points](https://review.udacity.com/#!/rubrics/432/view)

![alt text][cover]

#### Files Submitted & Code Quality

My project includes the following files:

**model.py** contains the script to create, train, validate and save the convolution neural network.  
`python model.py`

Using Udacity provided simulator and **drive.py**, the car can be driven in autonomous mode. **drive.py** can also generate *.jpg in a given folder. **model.h5** contains a trained convolution neural network.  
`python drive.py model.h5 auto_drive/`

**video.py** combines all images in a given folder to a mp4 file.  
`python video.py auto_drive/ --fps 48`

**auto_drive.mp4** shows how it drives for one lap on track No.1.  

## Model Architecture and Training Strategy

### 1. An overview of model architecture, dataset and hyperparameters

![alt text][model_architecture]


The model contains dropout layers in order to reduce overfitting (model.py lines 21)  

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). It was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I choosed and created the training data, see the next section.

### 2. Solution Design Approach

There are many topics to discuss in this section. I found out that it's hard to discuss them step by step since most of the time I tuned hyperparameters alternately. Therefore, discussion would be arranged topic by topic. Besides, information on Udacity's forum and cheatsheet is priceless. By following its suggestions, I dodged many pitfalls and unnecessary hard work. 

#### Model Architecture
I used a convolution neural network model similar to the Lenet-5. I thought this model might be appropriate because it can extract features out from an image. A mentor suggested that overfitting a small training set can help to find out the best model architecture. Therefore, I randomly picked about 3000 images from Udacity's dataset and fed them into CNN with different hyperparameters. After running each training process by 3 epochs, I picked the architecture which has lowest loss. After the architecture was determined, I applied dropout layer on fully connected layers to reduce overfitting on validation dataset. Additionally, it's worth to mention that validation loss could get larger while adding dropout layer and regulariztion simultaneously.

#### Preprocess Images
Preprocess images with the following steps (in the order of time sequence)
 - Converting images to RGB format. It can make the images' format being consistent with images in drive.py.
 - Normalize the images to 0 &le; x &le; 1.
 - Cropping the background out of images and left the tracks in the images.

![alt text][crop]

 - Resize images to be quarter of the original images.

![alt text][resize]

#### Data Collection and Selection

Driving along the dirt section of the track No.1 for about 5 to 10 times.

Augment flipped images and steering angles by executing
```python
images.append(cv2.flip(center_image, 1))
steering.append(center_steer*-1.0)
```
![alt text][center]
![alt text][center_flip]

Augment the training dataset by using left and right camera and correcting steering angle with a fixed value
```python
correction = 0.3
left_steer = center_steer + correction
right_steer = center_steer + correction
```
![alt text][left]
![alt text][right]

The proportion of training set to validation set is 3:1.

Augment the training dataset only. Do not augmenting the validation dataset.

Based on discussion from the Udacity's forum, too much images of car driving in a straight line of the road will make the model bias to the car driving in a straight line. In other words, the car couldn't drive well when facing curves. Thus, it's necessary to filter out the images when the car is driving in a straight line.  
Steering angle ranges from -1 to 1. Filtering out whose absolute value of the steering angle is less than 0.08 performs well.
```python
if abs(center_steer) >= 0.08:
	# Add images to list
```

#### Keras Version
At the begining, I used Keras 1. There is a parameter `samples_per_epoch` in `fit_generator()`. `samples_per_epoch` is a difficult number to decide, since I don't know how many samples I am going to feed into the model. Hence, I used Keras 2. `steps_per_epoch` replaces `samples_per_epoch`.

#### General Steps To Approach The Final Solution
**Step 1**: I used Udacity's dataset to train the CNN, but the car kept running out off the track. The driving didn't improve after many times of tuning on hyperparameters. Thus, I believe it's necessary to collect more data of driving along dirt side road.

**Step 2**: Driving along the dirt section on the track No.1 for about 5 times might be enough to let the model learn how to drive on the dirt section.

**Step 3**: After adding data, the training and validation loss became larger and the car still drove off the road. In addtion, some drunken behavior emerged in a straight line. It made me start to think about the CNN architecture was not complicate enough to make the prediction.

**Step 4**: By modifying the model architecture and hyperparameters, I attempted to overfit a small training dataset and tried to reduce the training loss at my best. After the architecture was found, adding dropout layers can overcome overfitting on validation dataset.

**Step 5**: After adding data and modifying the architecture, the car drove well on straight line. But it still drove off the road on dirt section of the road. However, the car started to turn except that the turning angle is not large enough to turn the corner.

**Step 6**: I suspected that the number of images of car driving in a straight are too much, which can bias the model to straight line. Thus, filtering out those images might reduce the bias. Filtering out the images whose steering angle are smaller than a threshold.

**Step 7**: The car now can drive autonomously without being off the track. Then, I used original Udacity's dataset with final architecture, hyperparameters and technic of data selection. It also works. : ) However, using addtional data can help the car driving smoothly.

