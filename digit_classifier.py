# import the argmax function from numpy to get the index of the maximum value in an array
from numpy import argmax
# import the mnist dataset from keras, which contains 60,000 images of handwritten digits for training and 10,000 images for testing
from keras.datasets import mnist
# import the to_categorical function from keras to convert integer labels to one-hot encoded vectors
from keras.utils import to_categorical
# import the load_img function from keras to load an image from a file
from keras.utils import load_img
# import the img_to_array function from keras to convert an image to a numpy array
from keras.utils import img_to_array
# import the load_model function from keras to load a saved model from a file
from keras.models import load_model
# import the Sequential class from keras to create a linear stack of layers for the model
from keras.models import Sequential
# import the Conv2D class from keras to create a convolutional layer that applies filters to the input image and produces feature maps
from keras.layers import Conv2D
# import the MaxPooling2D class from keras to create a pooling layer that reduces the size of the feature maps by taking the maximum value in each region
from keras.layers import MaxPooling2D
# import the Dense class from keras to create a fully connected layer that performs a linear transformation on the input vector and applies an activation function
from keras.layers import Dense
# import the Flatten class from keras to create a layer that flattens the input tensor into a one-dimensional vector
from keras.layers import Flatten
# import the SGD class from keras to create a stochastic gradient descent optimizer with a learning rate and a momentum parameter
from keras.optimizers import SGD
# import matplotlib.pyplot as plt to plot and show images using matplotlib library
import matplotlib.pyplot as plt
# import os.path to check if a file exists in the current directory
import os.path
# import sys to exit the program if an invalid input is given by the user
import sys

from sklearn.model_selection import KFold


# define the model file name as a global variable
model_file_name = 'mnist_cnn_test_1.h5'


# define a function to load and prepare the train and test dataset
def load_dataset():
    # load the mnist dataset using the load_data function from keras and assign the train and test data to four variables: trainX, trainY, testX, testY
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape the train and test images to have a single channel (grayscale) by adding a dimension of size 1 at the end of each array using the reshape method
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode the train and test labels using the to_categorical function from keras
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # return the four variables as output of the function
    return trainX, trainY, testX, testY


# define a function to scale the pixel values of the train and test images
def prep_pixels(train, test):
    # convert the train and test images from integers to floats using the astype method
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize the pixel values to range 0-1 by dividing them by 255.0
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return the normalized images as output of the function
    return train_norm, test_norm


# define a function to create and compile a CNN model
def define_model():
    # create an instance of the Sequential class and assign it to a variable named model
    model = Sequential()
    # add a convolutional layer with 32 filters of size 3x3, relu activation function, he_uniform weight initialization and input shape of 28x28x1 using the add method and passing an instance of the Conv2D class as argument
    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # add a max pooling layer with pool size of 2x2 using the add method and passing an instance of the MaxPooling2D class as argument
    model.add(MaxPooling2D((2, 2)))
    # add a convolutional layer with 64 filters of size 3x3, relu activation function and he_uniform weight initialization using the add method and passing an instance of the Conv2D class as argument
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform'))
    # add another convolutional layer with 64 filters of size 3x3, relu activation function and he_uniform weight initialization using the add method and passing an instance of the Conv2D class as argument
    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform'))
    # add another max pooling layer with pool size of 2x2 using the add method and passing an instance of the MaxPooling2D class as argument
    model.add(MaxPooling2D((2, 2)))
    # add a flatten layer to convert the output of the previous layer into a one-dimensional vector using the add method and passing an instance of the Flatten class as argument
    model.add(Flatten())
    # add a dense layer with 100 units, relu activation function and he_uniform weight initialization using the add method and passing an instance of the Dense class as argument
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # add another dense layer with 10 units (corresponding to the 10 classes of digits) and softmax activation function to output a probability distribution over the classes using the add method and passing an instance of the Dense class as argument
    model.add(Dense(10, activation='softmax'))
    # compile the model by specifying the optimizer, loss function and metrics using the compile method
    # create an instance of the SGD class with a learning rate of 0.01 and a momentum of 0.9 and assign it to a variable named opt
    opt = SGD(learning_rate=0.01, momentum=0.9)
    # use the opt variable as the optimizer argument, use categorical_crossentropy as the loss function for multi-class classification and use accuracy as the metric to evaluate the model performance
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # return the model as output of the function
    return model


# define a function to run the test harness for evaluating a model
def run_test_harness():
    # load and prepare the train and test dataset using the load_dataset function and assign them to four variables: trainX, trainY, testX, testY
    trainX, trainY, testX, testY = load_dataset()
    # scale the pixel values of the train and test images using the prep_pixels function and assign them to two variables: trainX, testX
    trainX, testX = prep_pixels(trainX, testX)
    # create and compile a cnn model using the define_model function and assign it to a variable named model
    model = define_model()
    # fit the model on the train dataset using the fit method with 10 epochs (number of iterations over the entire dataset), batch size of 32 (number of samples per gradient update) and verbose set to 1 (progress messages or 0 for not)
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('evaluate result > %.3f' % (acc * 100.0))
    # save the model to a file using the save method and passing the model file name as argument
    model.save(model_file_name)


# define a function to load and prepare an image for prediction
def load_image(filename):
    # load an image from a file using the load_img function from keras with grayscale set to True (convert to grayscale) and target_size set to (28, 28) (resize to match the input shape of the model) and assign it to a variable named img
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert the image to a numpy array using the img_to_array function from keras and assign it to a variable named img
    img = img_to_array(img)
    # reshape the image array to have a single sample with one channel by adding a dimension of size 1 at the beginning and at the end of the array using the reshape method and assign it to a variable named img
    img = img.reshape(1, 28, 28, 1)
    # the astype method and normalizing the pixel values to range 0-1 by dividing them by 255.0
    img = img.astype('float32')
    img = img / 255.0
    # return the image array as output of the function
    return img

# define a function to load an image and predict the class using the model


def run_example(path):
    # load and prepare the image using the load_image function and passing the path argument as filename and assign it to a variable named img
    img = load_image(path)
    # load the model from a file using the load_model function and passing the model file name as argument and assign it to a variable named model
    model = load_model(model_file_name)
    # predict the class of the image using the predict method of the model and passing the img variable as argument and assign it to a variable named predict_value
    predict_value = model.predict(img)
    # get the index of the maximum value in the predict_value array using the argmax function from numpy and assign it to a variable named digit
    digit = argmax(predict_value)
    # print the digit variable to show the predicted label
    print(digit)
    # plot and show the image using matplotlib.pyplot library
    # use the imshow function to display the image array (the first element of the img variable) with a grayscale colormap
    plt.imshow(img[0], cmap='gray')
    # use the title function to set a title for the image with 'Predicted label: ' followed by the digit variable
    plt.title('Predicted label: ' + str(digit))
    # use the show function to display the figure
    plt.show()


# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# ------------------ENTRY POINT------------------
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------


# ask the user if they want to re-train the data or use an existing model file using the input function and assign it to a variable named re_train
re_train = input('re train data and evaluate model? (0 -> false | 1 -> true): ')

# end program if input condition not satisfied by printing a message and using sys.exit function
if re_train != "0" and re_train != "1" and re_train != "":
    print("input condition not satisfied")
    sys.exit()


# check if a model file exists in the current directory using os.path.isfile function and if re_train is 0 or nothing using logical operators
if os.path.isfile(model_file_name) and (re_train == "0" or re_train == ""):
    # load model from file using load_model function and assign it to a variable named model
    model = load_model(model_file_name)
else:
    # run test harness to train and save a new model using run_test_harness function
    run_test_harness()


# run example to load an image and predict its class using run_example function with p as path argument
p = 'test_0_1.png'
run_example(path=p)