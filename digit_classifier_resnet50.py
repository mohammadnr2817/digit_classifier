import tensorflow as tf  # import tensorflow library as tf
import numpy as np  # import numpy library as np
from keras.utils import load_img  # import load_img function from keras utils
# import img_to_array function from keras utils
from keras.utils import img_to_array
# import load_model function from keras models
from keras.models import load_model
import matplotlib.pyplot as plt  # import matplotlib.pyplot library for plotting
import os.path  # import os.path module for file operations
import sys  # import sys module for system operations

model_file_name = 'mnist_resnet50_test_1.h5'  # define the name of the model file


def train_model():  # define a function to train the model
    # load the MNIST dataset and split it into training and testing sets
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0  # normalize the training images to [0, 1] range
    # add a channel dimension to the training images
    x_train = np.expand_dims(x_train, axis=-1)
    # repeat the channel dimension three times to match the ResNet50 input shape
    x_train = np.repeat(x_train, 3, axis=-1)
    # resize the training images to 32x32 pixels
    x_train = tf.image.resize(x_train, [32, 32])
    # convert the training labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    # print the shapes of the training images and labels
    print(x_train.shape, y_train.shape)

    # create an input layer with the shape of 32x32x3
    input = tf.keras.Input(shape=(32, 32, 3))
    base_model = tf.keras.applications.ResNet50(  # create a base model using ResNet50 architecture
        weights='imagenet', include_top=False, input_tensor=input)  # use imagenet weights and exclude the top classification layer
    # add a global average pooling layer after the base model output
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    # add a dense layer with 10 units and softmax activation for the final output
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    # create a model by connecting the input and output layers
    model = tf.keras.Model(inputs=input, outputs=output)

    base_model.trainable = False  # freeze the base model weights

    model.compile(  # compile the model with the following parameters
        # use categorical crossentropy as the loss function
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # use categorical accuracy as the metric
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam()  # use Adam as the optimizer
    )
    # fit the model on the training data with batch size of 128 and 10 epochs
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
    model.save(model_file_name)  # save the model to a file
    return model  # return the model


def load_image(path):  # define a function to load an image from a path
    # load the image and resize it to 32x32 pixels
    image = load_img(path, target_size=(32, 32))
    image = img_to_array(image)  # convert the image to a numpy array
    # add a batch dimension to the image array
    image = np.expand_dims(image, axis=0)
    # normalize the image to [0, 1] range
    image = image.astype('float32') / 255
    return image  # return the image


def predict_image(path):  # define a function to predict the label of an image from a path
    image = load_image(path)  # load the image using the load_image function
    prediction = model.predict(image)  # predict the label using the model
    # get the index of the highest probability in the prediction array
    label = np.argmax(prediction)
    print(label)  # print the predicted label
    # plot the image using matplotlib.pyplot library
    plt.imshow(image[0], cmap='gray')
    # add a title to the plot with the predicted label
    plt.title('Predicted label: ' + str(label))
    plt.show()  # show the plot


# ask the user if they want to retrain the model or not
re_train = input(
    're train data and evaluate model? (0 -> false | 1 -> true): ')

if re_train != "0" and re_train != "1" and re_train != "":  # check if the user input is valid or not
    # print an error message if not valid
    print("input condition not satisfied")
    sys.exit()  # exit the program

# check if there is a saved model file and if retrain is false or empty
if os.path.isfile(model_file_name) and (re_train == "0" or re_train == ""):
    model = load_model(model_file_name)  # load the model from the file
else:
    model = train_model()  # otherwise train a new model

p = '0.png'
predict_image(p)  # predict an image using its path