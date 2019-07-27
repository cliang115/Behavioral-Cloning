# **Behavioral Cloning** 

## Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./original.png "Center Image"
[image2]: ./left_image.png "Left Image"
[image3]: ./right_image.png "Right Image"
[image4]: ./flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 convolutional layers and two 3x3 convolutional layers. The depths are between 24 and 64 (model.py lines 59-63). The convolutional layers are followed by four fully connected layers. 

The model includes RELU layers to introduce nonlinearity (code line 59-63), and the data is normalized in the model using a Keras lambda layer (code line 57). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 64, 69, 71, 73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 54, 77). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the nVidia model. I thought this model might be appropriate because it is published and used to drive real car autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there are dropout layers (with a keep probability of 0.6) at the end of the convolutional layers and between the fully connected layers.

Then I examined the driving log csv file. There are much more occurrences of zero or near zero steering angles than large angles. I randomly dropped 92% of the images with zero steering angles. I also increased the occurrences of large steering angles by appending their images and angles more times. This helps with dealing with sharp turns.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I adjusted the offset angles of the images taken from the left and right cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 56-74) consisted of a convolution neural network with the following layers and layer sizes: a Keras Lambda normalization layer, three 5x5 convolutional layers and two 3x3 convolutional layers, a dropout layer and four fully connected layers with dropout layers in between. 

* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: relu
* Convolution: 5x5, filter: 36, strides: 2x2, activation: relu
* Convolution: 5x5, filter: 48, strides: 2x2, activation: relu
* Convolution: 3x3, filter: 64, strides: 1x1, activation: relu
* Convolution: 3x3, filter: 64, strides: 1x1, activation: relu
* Dropout(0.4)
* Fully connected: neurons: 120
* Dropout(0.4)
* Fully connected: neurons: 60
* Dropout(0.4)
* Fully connected: neurons: 12
* Dropout(0.4)
* Fully connected: neurons: 1

#### 3. Creation of the Training Set & Training Process

I used the images and data provided by Udacity including the images from the center, left and right cameras. Here is an example of center lane driving and recording from the left and right sides of the road:

![alt text][image1]

![alt text][image2]

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would help the model cope with both left and right turns. For example, here is the image that has then been flipped from the above center image:
![alt text][image4]

I then preprocessed this data by cropping out the sky and car front parts, resized to (320, 160) and converted to RGB. I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
