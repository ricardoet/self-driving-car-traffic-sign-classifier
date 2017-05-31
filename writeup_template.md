#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I simply read the pickled arrays to get the information I needed.
The summary of the data set is as follows:

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an example of an image contained in the set:

![alt text][image1]

Also, here is the histogram showing the image distribution for each of the 43 labels in the training set:

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th and 6th code cell of the IPython notebook.

As a first step, I decided to augment the training set to have a lower variance in the distribution of images per label. As I describe in the notebook, basing on the LeCun paper, rotating the images between -15 and 15 degrees as well as translating the images between -0.2 and 0.2 won't affect the neural network (it won't think it's a completely new traffic sign) so I augmented the set with a new image based on an image of the set with a random translation and rotation between the thresholds.

Here is the final histogram showing the new distribution after the augmentation.

Afterwards I decided to grayscale the images (based on the LeCun paper) and also normalized them as I read this helped. This can be found on the 6th code cell of the IPython notebook. This was done also on the test and validation sets (at first I didn't did it on all sets and lost days figuring out the issue :S).


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The sets were already splitted. We were given 3 pickled data files, one for each of the sets (train.p, valid.p, test.p) so I didn't had to do this myself.

As for the data augmentation as I said just above on point #2 I decided to augment the data to decrease the variance on the original distribution by generating new images based on the training images and changing them with a random rotation and translation.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding,  outputs 5x5x32 				|
| Fully connected		| Input: 800  Output: 400        									|
| RELU + 90% Dropout      |         |
| Fully connected		| Input: 400  Output: 120        									|
| RELU + 90% Dropout      |         |
| Fully connected		| Input: 120  Output: 43        									|
| 			|         									|
| Learning Rate    | 0.001     |
| Epochs      | 25      |
| Batch Size       | 100      |
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13th cell of the ipython notebook. 

To train the model, I used a learning rate of 0.001, 25 epochs and a batch size of 100. I don't really have a scientific reason for choosing this hyperparameters, I just find that was what worked best without taking a bunch of hours. 

For the optimizer I used the "Adam Optimizer" as was the default in the starting project files.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.984
* validation set accuracy of 0.955 
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was obviously the one from the lab, this yielded around 87% accuracy, pretty low for the >93% I was aiming for.

* What were some problems with the initial architecture?
As I said before, it could only get to aroung 87-88%, even changing the hyperparameters. It was clear that I had to pre-process the images and change the network's architecture in order to continue getting better accuracy.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Overall the whole setup was changed several times and in really big ways. I had trouble realizing what changes had to be made in order to get better predictions. At the end, the obvious worked, deeper and wider networks worked best. I think having 2 initial convolutional networks with 5x5 filters was an important part of the whole network.

Also, augmenting the dataset, changing it to grayscale and normalizing helped bring the validation accuracy to >95%.

* Which parameters were tuned? How were they adjusted and why?
Basically everything was tuned, either because I was overfitting the training set or because I couldn't reach 93% after hours of training on Amazon's AWS (which can get a bit pricey!). The ones that were changed but at last ended being the same were the learning rate and the batch size, which are pretty standard.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As I stated before, I think having 2 initial convolutional networks with big filters (5x5) is key to the success of my neural network. Also, having a dropout after each layer is pretty standard (left it at 0.9 as 0.8 took too much time to train and sometimes seemed to reach a local peak before, which makes no sense to me).

Also, making the neural network somehow deep is important too, being careful of not going to deep because the training time can grow really fast because of this, depending on your network structure.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<p align="center">
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/img1.png" width="150"/>
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/img2.png" width="150"/>
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/img3.png" width="150"/>
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/img4.png" width="150"/>
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/img5.png" width="150"/>
</p>

I was really careful choosing "clean" images (no watermark text or anything above the sign) as to not affect the neural network's capacity of predicting the correct label. I thought I could get a reasonable prediction accuracy on these images but as you'll read below it wasn't exactly the case.

One thing I did, which as far as I know it shouldn't affect the NN, was crop and rescale the images to fit the 32x32 needed size.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/hr      		| 20 km/hr   									| 
| No stopping     			| End of all speed and passing limits 										|
| Priority Road					| Priority Road											|
| 12-ton Weight Limit	      		| 120 km/hr					 				|
| 100 km/hr			| 20 km/hr      							|


The model was able to predict correctly only 1/5 images, giving an accuracy of 20%, really much lower than the 93% test accuracy. As I explain below, it wasn't so far off, predicting images that really look alike or share characteristics.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

<p align="center">
  <img src="https://github.com/ricardoet/self-driving-car-traffic-sign-classifier/blob/master/german%20signs/softmax%20probabilities%20new%20images.jpg" width="150"/>
</p>

As you can clearly see in this table showing the top 5 softmax probabilities for each of the new images, my NN was always "pretty sure" about its prediction, having 3 predictions at 100%, another one at 89% and the last one at 65%.

It's important to note that even though my NN only accurately predicted 1/5 images, either the first or second prediction of the failed ones are labels that look alike, for example predicting a 120km/hr label instead of a 12-ton limit.
