Traffic Sign Recognition
=========================== 

The goals / steps of this project are the following:
* Load the data set
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

## Writeup / README

#### This file includes all the [rubric points](https://review.udacity.com/#!/rubrics/481/view) and how I addressed each one.
#### Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x48    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48   				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 3x3x96      |
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 3x3x96   				|
| Flatten       		| outputs 1200  								|
| Fully connected 1200	| outputs 360  									|
| RELU					|												|
| Fully connected 360	| outputs 252  									|
| RELU					|												|
| Fully connected 252	| output 43  									|
| Softmax				| output 43    									| 

### 5. How I trained the model. This discussion includes the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Hyperparameters:
* Optimizer = AdamOptimizer
* Initial learning rate = 0.001
* Batch size = 128
* EPOCHS = 15

### 6. Description of the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

* I believed Lenet-5 architecture is capable of classifying digits as well as traffic signs. Lenet-5 classify digits by theirs shape. For that reason, traffic signs are just the images that have more complicate shapes than digits. Therefore, I believe that Lenet-5 is able to classify traffic signs by increasing the number of its neurons and layers.

* At the begining, I used original training dataset and default Lenet-5 architecture to train the model.  Digits and The validation accuracy is 0.89. 

* Traffic signs' complexity is higher than digit's. I believe that more number of neurons in conv layers can extract more complex features. Therefore, I tripled the size of filter and fully connected layer. The validation accuracy is improved but not enough to pass 0.93.

* I discovered that the distribution among types of signs is not uniform. It's necessary to augment the images which has fewer number in training dataset.

* After applying rotation, shear and adding uniform noise to selected images, the validation's accuracy is improved. However, the accuracy can not stably stay higher than 0.93. Thus, I decide to plot validation dataset's precision and recall to see which label keep dragging down the accuracy.

![alt text][valid_prec_recall]

![alt text][valid_f1_score]

* No.0, 24, 26, 27, 41 have really bad precision and recall. Therefore, I decide to add more augmented images of them. This time I still shear them but with different angles.

* The result didn't improved. In fact, it gets worse on some other labels too. I decide to add more neurons and add one extra conv layer.

* Now the validation accuracy can stably stay stay higher than 0.93.

My final model results were:
* training set accuracy is 0.998
* validation set accuracy is 0.962
* test set accuracy is 0.946

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
| Speed limit (70km/h)  | Speed limit (70km/h)   						| 
| Speed limit (20km/h)	| Speed limit (20km/h)							|
| Speed limit (30km/h)	| Slippery road   								|
| stop	      			| stop	      					 				|
| General caution		| General caution	   							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.6%.

![alt text][test_prec_recall]

As you can see, "Speed limit (30km/h)" is lable No.1, it has pretty good precision and recall. Thus there are only one possibility. Slippery road(No.23) has really bad precision.

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

## Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

###1. The visual output of my trained network's feature maps. What characteristics did the neural network use to make classifications?


