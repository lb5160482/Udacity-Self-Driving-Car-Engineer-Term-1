# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/image_distribution.jpg "Visualization"
[image2]: .//writeup_images/test_images.jpg "Traffic Signs"
[image3]: ./examples/random_noise.jpg "Random Noise"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and OpenCV library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of training set data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not convert the images to gray scale since I think the colors does influence the recognition. For example, some signs are with black edges while others are with red edges. Keeping three channels should be able to help increase the accuracy.

Next, I normalized the image data using
```python
def normalize(imgs):
    return (imgs / 256) - 0.5
```
so that all of the images has mean zero.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6 	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				   	|
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU 					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16					|
| Fully connected		| 400x120 weights, output dimension 120			|
| RELU 					|												|
| Fully connected		| 120x84 weights, output dimension 84			|
| RELU 					|												|
| Fully connected		| 84x42 weights, poutput dimension 42			|
| Softmax				| 	        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer to minimize the cross entropy loss. I used the batch size as 128 with 30 epochs. I also added dropout to the first two fully connected 
layer with the dropout rate 0.5. Also I used the mean value of 0 and standard deviation 0.1 to initialize the weights according to a normal distribution. After some times of tries, 
I selected the learning rate to be 0.00056.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.938
* test set accuracy of 0.912

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* I tried the LeNet because this architecture is proved to work well on image classification.
* What were some problems with the initial architecture?
* There is no dropout so at the beginging the it is not easy to make the validation set accuracy to be above 0.93.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* I added dropout to the first two fully connected layers. Without the dropout layer, the architecture is a little bit overfitting.
* Which parameters were tuned? How were they adjusted and why?
* I actually just decreased the learning rate because the default one 0.001 seems to be a little bit large, which cause the gradient descent bounded.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* A convolution layer is helpful to compute the image feature liki edges, corners and specific patterns. Using convolution layer will make use of the space information on the image. A simple neural network will only take pixel into consideration, which loses the space information of the image. Using convolution layer also let the whole image share the same weight(kernal), which greatly reduces the amount of variables to be trained. Dropout help reduce overfitting since dropout only randomly keeps some of the nodes working while training. This will prevent that some nodes being dorminant with the training set thus making all the nodes to be trained equally.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]

These five images are all easy to train because they are pretty regular. If the shape is distorted by the view or the lightness is really dark, it will be hard to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			         					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Speed limit (20km/h) 						| Speed limit (20km/h)							| 
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited 		|
| Slippery road								| Slippery road									|
| Go straight or right	      				| Go straight or right					 		|
| Roundabout mandatory						| Roundabout mandatory     						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is better than the accuracy on the test set of 0.912.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a Speed limit (20km/h) (probability of 1.0), and the image does contain a Speed limit (20km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| SSpeed limit (20km/h)   						| 
| 0.0     				| Speed limit (120km/h)							|
| 0.0					| Speed limit (30km/h)							|
| 0.0	      			| Speed limit (70km/h)					 		|
| 0.0				    | Speed limit (80km/h)      					|


For the second image, the model is pretty sure that this is a Vehicles over 3.5 metric tons prohibited (probability of 1.0), and the image does contain a Vehicles over 3.5 metric tons prohibited. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Vehicles over 3.5 metric tons prohibited  	| 
| 0.0     				| Speed limit (80km/h)							|
| 0.0					| Children crossing								|
| 0.0	      			| Speed limit (100km/h)				 			|
| 0.0				    | End of no passing								|

For the third image, the model is pretty sure that this is a Slippery road	 (probability of 1.0), and the image does contain a Slippery road	. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery road									| 
| 0.0     				| Dangerous curve to the right					|
| 0.0					| Dangerous curve to the left					|
| 0.0	      			| Ahead only							 		|
| 0.0				    | Turn left Ahead 		     					|

For the fourth image, the model is pretty sure that this is a Go straight or right (probability of 1.0), and the image does contain a Go straight or right (20km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Go straight or right   						| 
| 0.0     				| Keep right									|
| 0.0					| Ahead only									|
| 0.0	      			| Roundabout mandatory					 		|
| 0.0				    | End of all speed and passing limits      		|

For the fifth image, the model is pretty sure that this is a Roundabout mandatory (probability of 1.0), and the image does contain a Roundabout mandatory. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Roundabout mandatory  						| 
| 0.0     				| Speed limit (30km/h)							|
| 0.0					| Keep left										|
| 0.0	      			| Speed limit (120km/h)					 		|
| 0.0				    | Speed limit (100km/h)      					|