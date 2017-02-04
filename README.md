# CarND-Behavioral-Cloning
Udacity Self Driving Car - 3rd Project - Behavioral Cloning

#General overview
In the third project of the Udacity Self Driving Car Nanodegree program we should create a CNN in order to steer a car around a pre-defined road. 

#Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- Readme summarizing the results

#Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
python drive.py model.h5

#Submssion code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works

#Description of data process

#Data generation
For the project I generated two types of data sets: a training data set and a validation data set. The test data set consists of a real drive in the simulator. 
- The training data set comprises approx. 70.000 images (sum of images from center, right, left camera). For generating the data I drove four times in a normal way in the circuit. There were a few spots where the vehicle fell off the track and in order to compensate for that I enriched the data set by focussing on data within the curves. So I recorded my driving in the curves up to 5 times. Moreover, I drove two rounds backwards. This improved the driving behaviour so that at the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
- For the validation data set I drove 6 rounds in the circuit and used this data as validation data. 

#Data processing of training data
From the data there is three different kind of data available: images from the right, center or left camera. Following the approach a Nvidia team, an image was randomly chosen. In order to cope with the different right/ left perspectives an additional angle was added the steering angle. After the selection, the following steps were applied to each image:
- Normalisation: A conversion to HSV is used to ensure that the image have the spectrum. HSV is less sensitive with respect to different lights
- Shifted images: In order to get more augmented data the images were randomly shifted either to the left or right side. The steering angle was adapted respectively.
- Cropped image: The image was cropped in order to remove the noise from the background. By cropping the image we assume that each camera has a fixed position and a fixed perspective.
- Flipping of the image: In order to get a good distribution of the steering angle the images were randomly flipped and the steering angle was adapted accordingly.

#Data processing of validation data
For the validation set only images from the center camera are used. After the selection, the following steps were applied to each image:
- Normalisation to HSV
- Cropping of image
- Randomly flipping of image

#Description of CNN
For the CNN I applied 'Transfer Learning' and implemted the CNN from Nvidia. The CNN is described as follows:
- The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations. We use strided convolutions in the first three convolutional layers with a 2x2 stride and a 5x5 kernel and a non-strided convolution with a 3x3 kernel size in the last two convolutional layers. 
- We follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering.
Additionally, in order to reduce overfitting after each layer a drop out of 0.3 has been added to the model. 

#Running the CNN
In order to avoid to run out memory the Keras ImageGenerator was used which passed the data in batch sizes from n=128 to the CNN. 
That data contains a large amount of '0' steering angles. In order to cope with that images with a steeting angles between -.1 and .1 were randomly removed from the data. 
The model used an adam optimizer, so the learning rate was not tuned manually.
