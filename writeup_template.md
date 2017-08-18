# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop a simple model step by step.
First I started with the simpler Lt possible model, input and output. 
I testet the loading of the data and the batch processing.
On my notebook the batch processing run 20x slower than the all at once solution.
Keras has chosen automatically a batch size of 32, I changed it to 128. 256 lead to a kernel error on my machine.
After this was working I tried with the LeNet5 model and trained with the available data set.
The car drove in a small circle offside the road. 
Because it was recommended in the project description I started over with the Nvidia Pipeline.
I trained it withe the available dataset and added mirrored picutres and steering angles for curves (steering angle <>0).
This was a total of 9367 images, meaning 1332 images were added to the 8035. This shows that the network trained with this has a bias for straight roads.
Then I wrote a smallo tool for predicting a single angle for a single image.
The prediction were mostly wrong, so it was obvious why the simulator could drive the road. 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

I assume it was missing training data because the Nvidia model in general should work.
I read the the theory to check my implementation of the model. [https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/].
The implementation was correct, so I generated more data by adding the left and right camera. Because the images are "off the road" from that left/right perspective, the steering angles has to be adjusted. 
The car should correct it's driving back to the center when it sees an image off the centre. Therefore a correction to the steering angle has to be added. 
After trying different paramters I ended up with 8% correction factor.
Using 3 cameras generates 3x as much training data, with additonally flipping images for curves even more.
Now I realized that training many times is too time consuming and added loading and saving of a model.
To run more Epochs I decided to continue on AWS GPU cloud computer. This was the point were switched from All-In-One processing to processing  with a generator.


### Statistics for NNVIDIA model ###
angel correction constant +/- 15 degree for left and right camera

| No. of Cams | cropped | added flipped| total samples | epochs 	|Data set 		| train loss 	| valid loss 	| Description 				|  Failure at					 	| Comment 		|
|:-----------:|:-------:|:------------:|:-------------:|:----------:|:-------------:|:--------------|:--------------|:--------------------------|:---------------------------------:|:-------------:|    
 1   	      |   no 	|1332	       | 9367  			| 1			|	initial		|  0.0353 		| 0.0151  		|   On track for 10 sec. 	| first left curve  | Nvidia model can hold the track for some seconds
 3   	      |   no	|19747	       | 35084     		| 1			|	initial		|  0.0316 		| 0.0128  		|  On track for 90 sec. 		| first sharp left curve, hit marking	| big improvement with 3 cams
 3   	      |   yes   |11025	       | 28104     		| 3			|	added 2 rounds fwd. |  0.0119 		| 0.0154  		|   On track 90 sec. 		| first sharp left curve, hit marking	| 
 3			  |   yes   | 3105 		   | 12261    		| 4  		|   added 2 rounds ba	|	0.0275		    |			0.0507	|  On track for 		| ! 								|  
 
At this point I realized that the model is overfitting. (statistik_3cam_crop_4epoch_backwards.JPG)
I decided to set it up new with droupout layers.
Moreover I decided to add 5 degrees to the steering angle corrections because the off-center correction was too slow.


### Statistics for NNVIDIA model with dropouts ###
Angel correction constant +/- 20 degree for left and right camera

| No. of Cams | cropped | added flipped| total samples | epochs 	|Data set 		| train loss 	| valid loss 	| Description 				|  Failure at					 	| Comment 		|
|:-----------:|:-------:|:------------:|:-------------:|:----------:|:-------------:|:--------------|:--------------|:--------------------------|:---------------------------------:|:-------------:|    
 1   	      |   yes 	|1332	       | 9367  			| 1			|	initial		|  0.0353 		| 0.0151  		|   On track for 10 sec. 	| first left curve  | Nvidia model can hold the track for some seconds

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
