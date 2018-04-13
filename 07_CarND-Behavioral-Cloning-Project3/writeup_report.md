# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./writeup_images/model_structure.png "Model Structure"
[image2]: ./writeup_images/loss.png "loss"
[image3]: ./writeup_images/three_images.png "three"
[image4]: ./writeup_images/three_images_cropped.png "three cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 - A video recording of your vehicle driving autonomously at least one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 68-70) 

The model includes RELU layers to introduce nonlinearity (code line 68-70 and 72-73), and the data is normalized in the model using a Keras lambda layer (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The previoud model does not contain dropout layer since I have augmented the data very much which has already decreased overfitting a lot. And the result shows that the model does not overfit. However, since the rubric of this project asks to use dropout layer, I added two dropout layers and each after the first two fully connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 47). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left camera image and right camera image. With left camera image, I added an offset 0.1(code line 37) to let the car recover from being too left. With right camera image, I added an offset -0.1(code line 39) to let the car recover from being too right/

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional layer to extract features first and then use dense layer to get the correct streering angle.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because the three convolutional layers should be enough to get good feature of one input image. My second step was to use dense layer to get the steering angle output, this is normal when using dense layers coming after convolutional layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that no matter how I change my model structure, the loss will never be below 0.01 and the testing result is bad. After searching a lot on the forum, I finally realized that it was because I trained the data using BGR image read from OpenCV but when testing we are feeding RGB images to the model. So when preprocessing the data, I just change the image from BGR to RGB then I got a very good performance. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-74) consisted of three convolution neural network with the following layers and layer sizes (32, 3, 3), (64, 3, 3), (128, 3, 3) all with stride (3, 3).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I tried using Udacity's dataset. I tried adding some data generated by myself, but it did not prove to work well.

To augment the data sat, I also flipped images and angles as well as making use of left and right camera images(code line 31-39) thinking that this would increase the data size and also generalize the data(preventing overfitting).

![alt text][image3]

After the collection process, I had 144,654 number of data points. I then preprocessed this data by cropping (code line 66) and normalizing(code line 67).

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the image below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image2]