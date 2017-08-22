Traffic Sign Recognition
=========================== 

#### Overview

In this project, I will use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out your model on images of German traffic signs that I find on the web.

#### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

#### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

The goals / steps of this project are the following:
* Load the data
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[label_distribution]: ./fig_photo/label_distribution.png "Label Distribution"
[cmp_color_gray]: ./fig_photo/cmp_color_gray.png "Accuracy Comparison"
[color_gray]: ./fig_photo/color_gray.png " "
[label_distribution_aug]: ./fig_photo/label_distribution_aug.png
[augmented_image]: ./fig_photo/augmented_image.png
[cmp_normalize]: ./fig_photo/cmp_normalize.png
[test_f1_score]: ./fig_photo/Test_f1_score.png "F1 score of Testing Dataset"
[valid_f1_score]: ./fig_photo/Valid_f1_score.png "F1 score of Validation Dataset"
[test_prec_recall]: ./fig_photo/Test_prec_recall.png "Precision and Recall of Testing Dataset"
[valid_prec_recall]: ./fig_photo/Valid_prec_recall.png "Precision and Recall of Validation Dataset"
[accuracy_all_dataset]: ./fig_photo/accuracy_all_dataset.png "Accuracy of Dataset"
[test_from_web_00]: ./fig_photo/test_from_web_00.png "Traffic Sign 1"
[test_from_web_01]: ./fig_photo/test_from_web_01.png "Traffic Sign 2"
[test_from_web_02]: ./fig_photo/test_from_web_02.png "Traffic Sign 3"
[test_from_web_03]: ./fig_photo/test_from_web_03.png "Traffic Sign 4"
[test_from_web_04]: ./fig_photo/test_from_web_04.png "Traffic Sign 5"
[bar_test_from_web_0]: ./fig_photo/bar_test_from_web_0.png "Image 1 top 5 softmax probabilities"
[bar_test_from_web_1]: ./fig_photo/bar_test_from_web_1.png "Image 2 top 5 softmax probabilities"
[bar_test_from_web_2]: ./fig_photo/bar_test_from_web_2.png "Image 3 top 5 softmax probabilities"
[bar_test_from_web_3]: ./fig_photo/bar_test_from_web_3.png "Image 4 top 5 softmax probabilities"
[bar_test_from_web_4]: ./fig_photo/bar_test_from_web_4.png "Image 5 top 5 softmax probabilities"
[visualize_feat_conv1]: ./fig_photo/visualize_feat_conv1.png "Visualization of Convolution Layer 1"
[visualize_feat_conv2]: ./fig_photo/visualize_feat_conv2.png "Visualization of Convolution Layer 2"
[visualize_feat_conv3]: ./fig_photo/visualize_feat_conv3.png "Visualization of Convolution Layer 3"

## Writeup / README

#### This file includes all the [rubric points](https://review.udacity.com/#!/rubrics/481/view) and how I addressed each one.
#### Here is a link to my [project code](https://github.com/lionlai1989/auto_car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

### 1. Basic summary of the data set.

* The size of original training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

### 2. Exploratory visualization of the dataset.

![alt text][label_distribution]

As shown, the distribution across all labels are not distributed averagely in traing, validation, and test dataset. But in each label, the proportion of three set is consistent with other labels. According to the fact that uniform distributed training dataset can be beneficial to training process, it's essential to add more augmented data on the training dataset.

There are numbers of ways to do the image transformation.

* First, reflection and translation are not options here since the reflected and prolonged traffic signs would not appear in the real world. 

* Second, according to the advantage of convolutional neural network, it can extract the local feature of the sings no matter where it locates on the image. Therefore, it's not necessary to add rotated images. But I still add rotated images to the training dataset.

* Third, shooting a sign from different angles can result to sheared images, hence sheared images are inclueded.

* Fourth, adding uniform noises to original training dataset. Besides, fixing some overflow issues with the following code.

```python
X_train_gray[i][X_train_gray[i]>249] = 249 # 249 + 6 <= 255(uint8)
X_train_gray[i][X_train_gray[i]<6] = 6 # 6 - 6 >= 0(uint8)
ns_X_train.append(X_train_gray[i] + cv2.randu(noise,(-6),(6))[:, :, None])
```

* Fifth, after transforming images, there might be some blank space along the edges and corners. The color difference between the blank space and the original image can form a visible line which would affect the whole training process. Thus using the following code can blur the line.

```python
image[image == 0] = 64
```

After adding augmented images on the training dataset. Here is what the distribution looks like.

![alt text][label_distribution_aug]

---

## Design and Test a Model Architecture

### 1. Preprocess the image

A traffic sign image before and after grayscaling.

![alt text][color_gray]

As a first step, I decided to convert the images to grayscale because after a simple verificatioin, the color doesn't matter much in this classification task. 

![alt text][cmp_color_gray]

As you can see, the accuracy of color image is slightly better than grayscale image of 0.1 percent. But in the later experiments I still stick to grayscale image. There are two reasons of doing this. First, 0.1 is not a notable difference. Second, in this task, classifying traffic signs is usually classifying them by the shape of signs, not color. Jusk like how human recognize sign. Using colorful images may help to classify traffic sighs, but it is not the crucial factor that takes the effect.

### 2. Augment the image

Apply rotation, shear, and uniform noises to images.

![alt text][augmented_image]

### 3. Normalize the dataset to 0 &le; x &le; 1.

As a last step, I normalized the image data because normalizing data can reduce training time and help the algorithm reach the local or global minimum.

Basically, there are three ways to normalize training data.

* Normalized to 0 &le; x &le; 1.
* Rescaled to -1 &le; x &le; 1.
* Standardized.

Applying which normalized method to algorithms depends on purposes of the tasks and types of the features.

Here is a simple result using three normalized method respectively on the original training dataset with the basic LeNet-5 architecture.

![alt text][cmp_normalize]


As it might be seen, there is no significant difference between these three methods. However, I chose normalizing dataset to 0 &le; x &le; 1 in this task. An images is an array ranged from 0 to 255, thus it's intuitive to normalize the array to 0 &le; x &le; 1. Besides, I use Rectified Linear Unit as the activation function.

Rescaling the training dataset to -1 &le; x &le; 1 is another option. Yet it is believed that scaling to -1 &le; x &le; 1 is best suited with hyperbolic tangent activation function.

Standarization with mean and standard deviation is really not an good option in this task. The main reason might be that standariztion of an image sabotage the relationship of a pixel with its adjacent pixel. As a matter of fact, the result of using standarized training dataset is unreasonable and not consistent with the theory. When validating the model, I found that the result of using validation dataset's mean and std is better than using the training dataset's mean and std.

### 4. Here is what my final model architecture looks like (including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 3x3x96      |
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 3x3x96   				|
| Flatten       		| outputs 288  								|
| Fully connected 288	| outputs 480  									|
| RELU					|												|
| Fully connected 480	| outputs 336  									|
| RELU					|												|
| Fully connected 336	| output 43  									|
| Softmax				| output 43    									| 

### 5. How I trained the model. This discussion includes the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters:
* Optimizer = AdamOptimizer
* Initial learning rate = 0.001
* Batch size = 128
* EPOCHS = 15

### 6. Description of the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

* I believed Lenet-5 architecture is capable of classifying digits as well as traffic signs. Lenet-5 classify digits by theirs shape. For this reason, traffic signs are just the images that have more complicate shapes than digits. Therefore, I believe that Lenet-5 is able to classify traffic signs by increasing the number of its neurons and layers.

* At the begining, I used original training dataset and default Lenet-5 architecture to train the model. Digits and The validation accuracy was 0.89.

* Traffic signs' complexity is higher than digit's. I believe that more number of neurons in conv layers can extract more complex features. Therefore, I tripled the number of filters and fully connected layers. The validation accuracy was improved but not enough to pass 0.93.

* I observed that the distribution among types of signs was not uniform. It's necessary to augment the images which has fewer number in training dataset.

* After applying rotation, shear and adding uniform noise to selected images, the validation's accuracy was improved. However, the accuracy could not stably stay higher than 0.93. Thus, I decided to plot validation dataset's precision and recall to show which label keep dragging down the accuracy.

![alt text][valid_prec_recall]

![alt text][valid_f1_score]

* No.0, 24, 26, 27, 41 had really bad precision and recall. Therefore, I decided to add more augmented images of them. This time I still sheared them but with different angles.

* The result didn't improved. In fact, it got worse on some other labels too. For that reason, I decide to add more neurons and add one extra conv layer.

* Now the validation accuracy can stably reach higher than 0.93.

My final model results were:
* training set accuracy is 0.988
* validation set accuracy is 0.956
* test set accuracy is 0.938

![alt text][accuracy_all_dataset]
 
Here are the Precision and Recall of **test set**.

![alt text][test_prec_recall]

![alt text][test_f1_score]

## Test a Model on New Images

### 1. Here are five German traffic signs found on the web. For each image, I discuss what quality might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test_from_web_00]

![alt text][test_from_web_01]

![alt text][test_from_web_02]

![alt text][test_from_web_03]

![alt text][test_from_web_04]

### 2. Discussion of the model's predictions on new traffic signs and comparison of the prediction to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | General caution   					    	| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Speed limit (60km/h)	| Speed limit (60km/h)   						|
| stop	      			| Traffic signals	      		 				|
| General caution		| General caution	   							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 93.8%.

![alt text][test_prec_recall]

For the first web image, I intentionally make the size of the sign smaller than it normally should be when cropping the original image. When I was training the model, I once augmented the data with cropped images(zoom in). But the result got worse. Therefore, I believe the size of a sign in an image can influence the training and decode process. In this case, the size of the sign itself is too small. I believe it's the reason that make it hard to recognize by my model.

For the second web image, it looks easy to me to recognize.

For the third web image, it looks easy to me to recognize.

For the fourth web image, it looks easy to me to recognize. But my model recognized incorrectly. The precision and recall of stop sign are pretty good. For traffic signals, the precision and recall are not so good but not the worst. The character of stop looks a bit of jittered to me. Perhaps it's the reason for incorrect recognization.

For the fifth web image, it looks easy to me to recognize.

### 3. Here are the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image

![alt text][bar_test_from_web_0]

For the second image 

![alt text][bar_test_from_web_1]

For the third image

![alt text][bar_test_from_web_2]

For the fourth image 

![alt text][bar_test_from_web_3]

For the fifth image 

![alt text][bar_test_from_web_4]

## Visualizing the Neural Network

### 1. The visual output of my trained network's feature maps. What characteristics did the neural network use to make classifications?

Image from web was fed into the network. Here are what they look like.

Visualization of conv1

![alt text][visualize_feat_conv1]

Visualization of conv2

![alt text][visualize_feat_conv2]

Visualization of conv3

![alt text][visualize_feat_conv3]

To be honest, conv2 and conv3 are confusing to me. I can't see how they can be used for later classification. Amazing CNN!!!
