# **Behavioral Cloning** 

## Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/console-loss.jpg "console loss"
[image2]: ./images/loss.png "loss plotting"
[image3]: ./images/driving.gif "driving example"
[image4]: ./images/left.jpg "left"
[image5]: ./images/center.jpg "center"
[image6]: ./images/right.jpg "right"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I decided to use a modified version of the Nvidia model, basically simplifying it to reduce the number of features and allow the network to train faster (with quite good results).
The model consists of the following layers:

- Cropping2D: input 160x320x3 and output 40x320x3 (removing 65 pixes from the top and 25 from the bottom) - AveragePooling2D with a 1x2 path: output 40x160x3 in order to reduce the following layers size
- Lambda min-max: Normalize the data with a min-max function (this way too dark or too bright images still get properly normalized)
- Convolution2D 8x8x24 with 2x2 stride, RELU activation
- Convolution2D 8x8x36 with 2x2 stride, RELU activation
- Convolution2D 5x5x48 with 2x2 stride, RELU activation 
- Convolution2D 3x3x64, RELU activation
- Convolution2D 3x3x64, RELU activation
- Flatten
- Dropout 0.5
- Dense 70, LINEAR activation
- Dropout 0.5
- Dense 30, LINEAR activation
- Dropout 0.5
- Dense 10, LINEAR activation
- Dense 1

I used 8x8 patches for the first 2 layers because I wanted to reduce the number of trainable parameters, using a larger patch wouldn't work because after the Cropping and the AveragePooling2D the image is already small enough.

The number of features from the convolutional layers remans the same than the ones from the Nvidia paper, all of them with RELU activation to add nonlinearity

Finally I added 3 fully conected layers, with LINEAR activation, but added dropout before each one of them.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout before all the fully connected layers in order to reduce overfitting.

I achieved a validation loss of 0.0051 and plotting the
traing and validation loss shows that, while there is still overfitting, the model is good enough to generarize for the 1st track

![alt text][image1]

![alt text][image2]


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I followed the advices given in the exercice and for the trainign data I took images of 1 lap in one direction and another lap in the opposite direction, also I took recovery examples, and took some more data in curves going specially smooth.
I also decided to take more examples in the bridge, as it was a section very different from the rest and I did not want the model to learn how to drive everywhere except there, and also more recovering in some specific curves where the lines are different (like dirt) so the model doesn't drive away in those parts.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started by following the examples from the exercice, first a simple fully connected multilayer network, then I implemented the LeNet and finally used the Nvidia model, that worked the best but was trainign very slowly.

I thought that, if in the traffic sign classification the model was working well enough with 32x32 images, this model might also work with a smaller images that's why I resized the image with to the half of the original one.

I also thought that with the image resized maybe the net didn't need so small patches for the initial features, so I increased the initial patches from 5 to 8.

Finally I decided to make the fully connected layers smaller, as I don't need the model to drive everywhere, just in the simulator, so the number of features I might need is smaller.

I recorded videos of all of those attempts, the model wasn't able to drive a full lap until I used the Nvidia model, after that I did some more finetunning until I achieved the model I explained.

Here's an example of thow the model manages to drive (the full video can be seen in the file video.mp4):

![alt text][image3]


#### 2. Final Model Architecture

I already explained the full model architecture in the section before.

#### 3. Creation of the Training Set & Training Process

As stated before:

I followed the advices given in the exercice and for the trainign data I took images of 1 lap in one direction and another lap in the opposite direction, also I took recovery examples, and took some more data in curves going specially smooth.

I also decided to take more examples in the bridge, as it was a section very different from the rest and I did not want the model to learn how to drive everywhere except there, and also more recovering in some specific curves where the lines are different (like dirt) so the model doesn't drive away in those parts.

Here are 3 samples of images taken: left, center, right

I used this 3 images applyting a correction of -0.15 to the left image and +0.15 to the right image to help the model learn how to be centered.

![alt text][image4]
![alt text][image5]
![alt text][image6]

I decided to not take any data from track 2, only data from track 1 to make the model work as good as possible on this one.

I realized that I should have taken more examples driving on straight roads, because the model seems to have learned with a bias on turns, and in straight lines it wobbles a little bit from right to left and back.

I trained the model with 30 epochs, only saving the model on the best validation loss, I achieved the best result after 20 epochs. 

The rest of the epochs resulted on overfitting, reducing the training loss but not the validation loss.

Overall the result is very acceptable.