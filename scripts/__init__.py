# ******************************** dataset prep script **********************************
# pull in labelled data and augment it by flipping everything vertically and horizontally
# data source: https://www.kaggle.com/pnkjgpt/butterfly-classification-dataset
'''
The walk() method of the os module also takes the directory path string as input and returns the path of the root directory as a string,
the list of the subdirectories, and the list of all the files in the current directory and its subdirectories.

To find the file with the name filename.txt, we can first get all the files in the directory and then loop through them
to get the desired file. The below code example demonstrates how to find a file by looping through the files in a directory.

import os

myfile = 'filename.txt'
for root, dirs, files in os.walk("Desktop/myFolder"):
    for file in files:
        if file == myfile: if file is photo,
            you can get the file path of the photo using: os.path.abspath(file)
            then create new photo of it, and save that new photo in the same file. w/ filename + flip

When this is complete we need to resize the images, I used openCV here. Be careful when using openCV because by default
openCV translates RGB images to BGR. In image classification this can lead to undesirable results, because certain
colors could help identify different classes. After you use openCV to adjust an image, you should always color correct.
'''

# Additional functions
'''
I added functions for removing the flipped images of data because there may be occasion when you make a mistake or you
want to go back to your original data set. This function should put you back at square one.

There is also a function to prep additional images for predictions. You can simply create a for loop if you have a list
of images and then run that function on each image. It does include color correction and resizing. At the time of this 
note, the model is strictly in script form, so there may be additional instructions once it is ready for deployment. 
'''

# ********************************* building the image classification model ****************************************
'''
 We're using a convolutional neural network here because I think they're ideal for image classification. At a high 
 level, the layers within a CNN are designed to examine smaller sections of the image and compare them to parts of 
 another image to see if there are any similarities. Each layer can focus on a different aspect of the image, eventually 
 piecing all the parts together to create a "feature map" that represents each image. Once that is done, these maps
 can be passed through a dense neural network to determine the class it actually belongs to.
'''

# As this is an example of supervised learning, we'll need to first label our data.
'''
I opted for just using part of the path name as the label. I saved the image  
'''


# ********* build model script *********
# add ability to log to a file and STout
# first, read in csv (pandas), or create DF with image path and labels, and convert the images to numpy arrays
'''
https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/
We'll be using the pillow (PIL) image library for this
'''
# then we will need to build a neural network, in this case I opt for a CNN. Since we're dealing with images, the
# layered approach of a CNN will probably perform better than a basic NN. We'll need TensorFlow/Keras for this
# split labelled data into train, validate, test sets
# train model on training set
# evaluate the model on validate set, and tweak
# finally run through test data to get final results

# ********* project wrap up *************
# create requirements document
# push to github
# create readme
# package model and deploy either via flask or via docker
# check for jpeg input then run prediction function from model
# output result in some kind of gui
