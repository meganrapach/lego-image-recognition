# Overview

This repo contains code for the "TensorFlow for poets 2" series of codelabs.

There are multiple versions of this codelab depending on which version 
of the tensorflow libraries you plan on using:

* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).


This repo contains simplified and trimmed down version of tensorflow's example image classification apps.

* The TensorFlow Lite version, in `android/tflite`, comes from [tensorflow/contrib/lite/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite).
* The Tensorflow Mobile version, in `android/tfmobile`, comes from [tensorflow/examples/android/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).

The `scripts` directory contains helpers for the codelab. Some of these come from the main TensorFlow repository, and are included here so you can use them without also downloading the main TensorFlow repo (they are not part of the TensorFlow `pip` installation).

# Problem Description

LEGO is famous for selling sets of bricks that can be built into well-known objects. For example, they have sets for certain car models, buildings, and elements from popular movies such as Hogwarts Castle from the Harry Potter series. These sets contain many bricks - large LEGO sets can be comprised of up to 75,000 pieces. 

Within a LEGO set, the bricks are usually separated into smaller bags. When these bags are opened, the builder is left with piles and piles of LEGO bricks. On top of being overwhelming, it can be very difficult to find a needed piece in this sea of LEGO pieces. Often, this can be demotivating to builders and rather straining on the mind and body (eyes, neck, etc.).

For our final project, we attempt to find a solution to this problem using image recognition to recognize and classify LEGO bricks based on color and size. This would help alleviate the stress faced by LEGO enthusiasts when searching for needed LEGO pieces in a large pile of bricks.

# Approach

As a proof of concept, we trained our image recognition model with images of 4 different color LEGO bricks of 3 sizes. 

* Colors - Red, Yellow, Green, Blue
* Sizes - 2x2, 2x3, 2x4

## Tools & Technologies

For our project, we used several tools to implement our algorithm. These tools include TensorFlow, Transfer Learning, and TensorBoard.
TensorFlow is a free, open-source machine learning tool that contains serval libraries and other resources for numerical computation machine learning projects. Popular use cases for TensorFlow include voice and sound recognition, text-based analysis applications, image recognition, time series analysis, and video detection. For our project, we use TensorFlow to train a classifier to classify our LEGO bricks by color and size.

Transfer Learning refers to a machine learning method in which a model that has been previously developed is reused as the starting point for a new model. Transfer Learning is a common approach to deep learning, which uses a pre-trained model as a starting point to more quickly develop neural networks to be used on related problems. We used a previously trained model for our LEGO image recognition called ImageNet. ImageNet was already pre-loaded with over 1,000 classes. We added our 12 new classes (one for every combination of color and size LEGO brick) and retrained the model. We then used a neural network called MobileNet to re-train ImageNet into a new model to compare only the new classifiers we had specified from our sample data of LEGO bricks.

To visualize the performance of our image recognition model, we used TensorBoard. TensorBoard is a dashboard that shows a graphical representation of the accuracy and cross entropy of the learning model. This tool was very useful because it allowed us to see how our re-training of the existing model was affected by the addition of our new classes, as well as see how well our model performed in terms of accuracy.

## Data Collection

In order to re-train the ImageNet model, we had to provide our own images of the LEGO bricks. We used the photo burst feature on our iPhones to take several pictures of each classifier, then made a directory for each classifier and placed the images in the corresponding folder. We attempted to make the images of each classifier as similar as possible, using the same back-drop (a white sheet of printer paper) and taking the pictures from similar angles at roughly the same distance from each brick. In total, we took around 3,100 images of LEGO bricks!

# Tutorial For Use

This section provides step-by-step instructions for replicating our project implementation. Note that this can be done for any set of new classifiers, as long as the images are stored correctly under the tf_files directory.

Our tutorial uses Linux shell commands, so it is recommended to follow this tutorial using a Linux machine, if possible.

The first thing we must do is install TensorFlow on our machine. This can be done by running the following command.

`pip install --upgrade "tensorflow==1.7.*"`

Next, we clone the git repository that contains the required python scripts.

`git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2`

After we have cloned the git repository, we can add our images that we will use to retrain the model to the tf_files folder. First, create a directory under tf_files to contain all the training images. We call this lego_photos. Under lego_photos, create a subdirectory for each of the 12 classifiers. The file structure for tf_files should look like this:

`tf_files/
   lego_photos/
      Blue_2x2/
      Blue_2x3/
      Blue_2x4/
      Green_2x2/
      Green_2x3/
      Green_2x4/
      Red_2x2/
      Red_2x3/
      Red_2x4/
      Yellow_2x2/
      Yellow_2x3/
      Yellow_2x4/`
      
To retrain the model with our LEGO dataset, we first must configure and retrain the MobileNet neural network. There are two variables we must set to configure the MobileNet.
* Input image resolution (128, 160, 192, or 224 px). Note that using a higher image resolution will take more processing time, but results in better classification accuracy.
* The relative size of the model as a fraction of the largest MobileNet (1.0, 0.75, 0.50, or 0.25)

We will use 224 px and 0.25 for this project.

These variables can be set in the Linux shell:

`IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.25_${IMAGE_SIZE}"`

Before we retrain the model, start TensorBoard in the background to monitor the training progress. To run TensorBoard, open a new Linux shell and run the following command.

`tensorboard --logdir tf_files/training_summaries &`

Note that the above command will fail if there is already an instance of TensorBoard running. If this is the case, the running instance can be terminated by running

`pkill -f "tensorboard"`

Now, we can retrain the neural network. Because we cloned the git repository from codelab, we already have all of the scripts required to do the retaining. All we have to do is run a python script with the settings we would like. Run the following in the Linux shell:

`python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/lego_photos`

This script only retrains the final layer of the ImageNet network by downloading the pretrained model, adding a new final layer, and training that layer on the LEGO images we added. While this step may take a while, it should complete in a reasonable amount of time.

Even though the original ImageNet does not contain any of the LEGO pieces we are using to train, the ability to differentiate among 1,000 classifiers is useful in distinguishing between objects. By using this model, we are taking that information as input to the final classification layer that distinguishes our LEGO classifiers.

Note that in the above command we run the script with only 500 training steps. Accuracy can be improved if this number is increased. The default value for this parameter is 4,000 steps.

The retrain script writes data to two files:
* tf_files/retrained_graph.pb – contains a version of the selected network with a final layer retrained on our classifiers
* tf_files/retrained_labels.txt – a .txt file containing labels

Now we are ready to test our model with a test image of a LEGO brick. First, we create a new directory under tf_files called test_images to hold the photos we will use to test. We create subdirectories for each classifier to keep our test images organized. The file structure will resemble that of the training folder. We have produced it below for reference.

`tf_files/
   test_images/
      Blue_2x2/
      Blue_2x3/
      Blue_2x4/
      Green_2x2/
      Green_2x3/
      Green_2x4/
      Red_2x2/
      Red_2x3/
      Red_2x4/
      Yellow_2x2/
      Yellow_2x3/
      Yellow_2x4/`
      
As an example, save the following image as IMG_6852.jpg to tf_files/Red_2x2

![Image of Red 2x2](/tf_files/test_images/Red_2x2/IMG_6852.jpg)

Once the image is saved to the aforementioned directory, we can run the label_image.py script to test the model’s ability to classify this image as a Red 2x2 LEGO brick. Run the following command to classify the image.

`python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/test_images/Red_2x2/IMG_6852.jpg`

Each execution of this script will print a list of LEGO labels, usually with the correct LEGO classifier on top. Results may look like the following.

`Evaluation time (1-image): 0.383s

Red 2x2 (score = 0.97167)
Yellow 2x2 (score = 0.02371)
Red 2x3 (score = 0.00324)
Yellow 2x3 (score = 0.00133)`

This result indicates that the model predicted with ~97% accuracy that the image we selected is a Red 2x2 LEGO brick, which is the correct classification.

To test more images, upload additional test images and run label_image.py again with the corresponding path to the image.

Red 2x4 (score = 0.00002)
