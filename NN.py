import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow import keras
def test (x):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Seems like it is: {np.argmax(prediction)}')

choice = str(input("\nHello.\nDo you want to compile new model?\n"))
if choice == "y":
    #Download dataset
    mnist = tf.keras.datasets.mnist

    # #Split dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Normalization
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    #Creating model
    model = tf.keras.models.Sequential()

    #Adding layers
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units = 96, activation =tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = 64, activation =tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = 96, activation =tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = 10, activation =tf.nn.softmax))

    #Compile Model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])

    #Model Fit
    model.fit(x_train,y_train, epochs = 15)
    
    choice = str(input("Want to save model?\n"))
    if choice == "y":
        model.save('./Model_NN')
    else:
        pass
elif choice == "n":
    choice = str(input("Want to load the model?\n"))
    if choice == "y":
        model = keras.models.load_model('./Model_NN')
    else:
        pass


choice = str(input("Want to test the model?\n"))
if choice == "y":
    test(int(input("Please Enter test picture name\n")))
else:
    print ("Then Goodbye")