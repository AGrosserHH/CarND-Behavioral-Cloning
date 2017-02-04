# CarND-Behavioral-Cloning
Udacity Self Driving Car - 3rd Project - Behavioral Cloning

#General overview
In the third project of the Udacity Self Driving Car Nanodegree program we should create a CNN in order to steer a car around a pre-defined road. 

#Data generation
For the project I generated two types of data sets: a training data set and a validation data set. The test data set consists of a real drive in the simulator. 
- The training data set comprises approx. 70.000 images (sum of images from center, right, left camera). For generating the data I drove four times in a normal way in the circuit. In order to have more variance in data I enriched the data set by focussing on data within the curves. So I recorded my driving in the curves up to 5 times. Moreover, I drove two rounds backwards.
- For the validation data set I drove 6 rounds in the circuit and used that data as validation data. 

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
For the CNN I applied and implemted the CNN from Nvidia. The CNN is described as follows:
- The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations. We use strided convolutions in the first three convolutional layers with a 2x2 stride and a 5x5 kernel and a non-strided convolution with a 3x3 kernel size in the last two convolutional layers. 
- We follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering.
Additionally, in order to reduce overfitting after each layer a drop out of 0.3 has been added to the model.

#Running the CNN
In order to avoid to run out memory the Keras ImageGenerator was used which passed the data in batch sizes from n=128 to the CNN. 
That data contains a large amount of '0' steering angles. In order to cope with that images with a steeting angles between -.1 and .1 were randomly removed from the data. 
