import tkinter as tk
import tkinter.ttk as ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

# topic 5
def show_multiple_images_and_labels(images, labels):
    plt.gcf().set_size_inches(12, 14)
    id = np.random.randint(50000, size=10)
    label = {'[0]':'airplane', '[1]':'automobile', '[2]':'bird', '[3]':'cat', '[4]':'deer',
           '[5]':'dog', '[6]':'frog', '[7]':'horse', '[8]':'ship', '[9]':'truck'}
    num = len(id)
    for i in range(num):
        show_image = plt.subplot(2, 5, i + 1)
        show_image.imshow(images[id[i - 1]], cmap='binary')
        r = labels[id[i - 1]]
        title = 'label = ' + str(label[str(r)])
        show_image.set_title(title, fontsize=12)
        show_image.set_xticks([])
        show_image.set_yticks([])
        i += 1
    plt.show()


def show_the_images():
    # input the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # show images
    show_multiple_images_and_labels(x_train, y_train)


# 5.1 show train image
def show_the_input_image():
    # import data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # plot the images
    show_multiple_images_and_labels(x_train, y_train)


# 5.2 hyperparameters
def hyperparameter():
    # set the hyperparameters
    epochs = 50
    batch_size = 32
    weight_decay = 0.0005
    learning_rate = 0.001
    optimizers = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    return epochs, batch_size, weight_decay, learning_rate, optimizers


def show_hyperparameter():
    epochs, batch_size, weight_decay, learning_rate, optimizers = hyperparameter()

    print("hyperparameters: ")
    print("batch size: " + str(batch_size))
    print("learning rate: " + str(learning_rate))
    print("optimizers:  SGD")
    print("epochs: " + str(epochs))


# 5.3 show_the_structure_of_model
def model():
    # import data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # set the data type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # set the one-hot vector
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # normalize th3 data and fix the shape of array
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    X_train = (x_train - mean) / (std + 1e-7)
    X_test = (x_test - mean) / (std + 1e-7)

    # set the hyperparameters
    epochs, batch_size, weight_decay, learning_rate, optimizers = hyperparameter()

    # training model: VGG16
    model = Sequential()

    # layer1: convolution
    # depth:64, filter:3x3, input: 32*32*3, activation = 'relu'
    model.add(Conv2D(64, (3, 3),
                     padding='same', input_shape=(32, 32, 3),
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # layer2: convolution and maxpooling; pool filter: 2x2
    # depth:64, filter:3x3, input: 32*32*64, activation = 'relu'
    model.add(Conv2D(64, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer3 16*16*64
    model.add(Conv2D(128, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer4 16*16*128
    model.add(Conv2D(128, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer5 8*8*128
    model.add(Conv2D(256, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer6 8*8*256
    model.add(Conv2D(256, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer7 8*8*256
    model.add(Conv2D(256, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer8 4*4*256
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer9 4*4*512
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer10 4*4*512
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer11 2*2*512
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer12 2*2*512
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # layer13 2*2*512
    model.add(Conv2D(512, (3, 3),
                     padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # layer14 1*1*512
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay),
                    activation='relu'))
    model.add(BatchNormalization())

    # layer15 512
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay),
                    activation='relu'))
    model.add(BatchNormalization())

    # layer16 512
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    # the following code is starting training a model and plot the result
    # if you want to implement them, just delete the green "

    """
    # set the loss function, optimizers and
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers,
              metrics=['accuracy'])

    history = model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.2,
          verbose=1)

    # save models
    model.save('cifar10.h5')

    # save models weights
    model.save_weights("cifar10_weights.weight")

    history.history.keys()

    # plot the loss and accuracy

    plt.plot(history.history['loss'], label = 'train')
    plt.plot(history.history['val_loss'], label = 'test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc = 'upper right')

    plt.show()

    plt.plot(history.history['accuracy'], label = 'train')
    plt.plot(history.history['val_accuracy'], label = 'test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')

    plt.show()
    """


def show_accuracy():

    img1 = cv2.imread('Accuracy.png')
    img2 = cv2.imread('Loss.png')

    imgs = np.hstack([img1, img2])
    cv2.imshow("Accuracy and Loss", imgs)
    cv2.waitKey(0)

def print_the_prediction_result(image, label, prediction, id):
    plt.gcf().set_size_inches(12, 14)
    show = plt.imshow(image[id], cmap = 'binary')
    # print the prediction result from ai
    label_dic = {'[0]': 'airplane', '[1]': 'automobile', '[2]': 'bird', '[3]': 'cat', '[4]': 'deer',
             '[5]': 'dog', '[6]': 'frog', '[7]': 'horse', '[8]': 'ship', '[9]': 'truck'}
    label_pre = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', '4': 'deer',
                 '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', '9': 'truck'}
    plt.show()
    print('prediction: '+str(label_pre[str(prediction[id])]))
    print('label: '+str(label_dic[str(label[id])]))
def plot_predict(image, label, Predicted_Probability, id):
    label_index = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    pp = np.reshape(Predicted_Probability[label[id], :], 10)
    plt.bar(label_index, pp, width=0.5)
    plt.xticks(rotation='vertical')
    plt.show()

def test(*args):
    # import data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # unfinished
    model = load_model('cifar10.h5')
    id = np.int32(input_image_index.get())
    prediction = np.argmax((model.predict(x_test)>0.5).astype("int32"), axis = -1)
    Predicted_Probability = model.predict(x_test)
    print_the_prediction_result(x_test, y_test, prediction, id)
    plot_predict(x_test, y_test, Predicted_Probability, id)


# build a window
window = tk.Tk()
window.title("Homework 1")
# window's size
window.geometry("270x270")

# 5.1 show train image
show_the_input_image = tk.Button(window,
                        text='5.1 show_the_input_image',
                        width = 30,
                        height = 2,
                        padx=1,
                        pady=1,
                        command = show_the_input_image ).pack()

# 5.2 show hyperparameters
show_hyperparameter = tk.Button(window,
                        text='5.2 show_hyperparameter',
                        width = 30,
                        height = 2,
                        padx=1,
                        pady=1,
                        command = show_hyperparameter ).pack()

# 5.3 show model structure
show_the_structure_of_model = tk.Button(window,
                        text='5.3 show_the_structure_of_model',
                        width = 30,
                        height = 2,
                        padx=1,
                        pady=1,
                        command = model ).pack()

# 5.4 show accuracy
show_accuracy = tk.Button(window,
                        text='5.4 show_accuracy',
                        width = 30,
                        height = 2,
                        padx=1,
                        pady=1,
                        command = show_accuracy ).pack()

# 5.5 test

test = tk.Button(window,
                        text='5.5 test',
                        width = 30,
                        height = 2,
                        padx=1,
                        pady=1,
                        command = test ).pack()

# input the test image index
image_index = tk.Label(window, text = '    Input the text : ', font = ('Arial', 12),
                    width = 10, height = 1)
image_index.pack(side = tk.LEFT)
input_image_index = tk.StringVar()
input_image_index_ = tk.Entry(window, textvariable = input_image_index, bd = 4 )
input_image_index_.insert(1, "0-9999")
input_image_index_.pack(side = tk.LEFT)
rotation_value = input_image_index.get()

# open the window
window.mainloop()