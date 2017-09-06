import os
from os.path import dirname, realpath

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
import matplotlib.pyplot as plt

PROJECT_DIR = dirname(dirname(realpath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


def build_model(batch_size=128, num_classes=10, epochs=10, layers=1, activation_fn='relu', add_dropout=False,
                kernel_size=(3, 3), no_of_kernels=32, dense_layer=True, l2_reg=0.0):

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(no_of_kernels, kernel_size=kernel_size,
                     activation=activation_fn,
                     input_shape=input_shape, kernel_regularizer=l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if layers >= 2:
        model.add(Conv2D(64, kernel_size, activation=activation_fn, kernel_regularizer=l2(l2_reg)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    if add_dropout:
        model.add(Dropout(0.2))
    model.add(Flatten())
    if dense_layer:
        model.add(Dense(128, activation=activation_fn))
        if add_dropout:
            model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history, score


def test_no_of_layers(plot_type='layers'):
    h1, s1 = build_model(layers=1, dense_layer=False)
    h2, s2 = build_model(layers=1)
    h3, s3 = build_model(layers=2)

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.plot(h3.history['val_acc'])
    plt.title('Model accuracy based on no. of hidden layers')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = '1 Convolution layer(Test accuracy - {})'.format(s1[1])
    legend2 = '1 Convolution layer, 1 dense layer(Test accuracy - {})'.format(s2[1])
    legend3 = '2 Convolution layers, 1 dense layer(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_cnn_{}.png'.format(plot_type))
    plt.close()


def test_activation_fun(plot_type='activation'):
    h1, s1 = build_model(activation_fn='relu')
    h2, s2 = build_model(activation_fn='sigmoid')
    h3, s3 = build_model(activation_fn='tanh')

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.plot(h3.history['val_acc'])
    plt.title('Model accuracy based on type of activation function')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = 'ReLU(Test accuracy - {})'.format(s1[1])
    legend2 = 'Sigmoid(Test accuracy - {})'.format(s2[1])
    legend3 = 'Tanh(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_cnn_{}.png'.format(plot_type))
    plt.close()


def test_kernel_size(plot_type='kernel_size'):
    h1, s1 = build_model(kernel_size=(3, 3))
    h2, s2 = build_model(kernel_size=(5, 5))
    h3, s3 = build_model(kernel_size=(10, 10))

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.plot(h3.history['val_acc'])
    plt.title('Model accuracy based on kernel size')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = '3x3 kernel(Test accuracy - {})'.format(s1[1])
    legend2 = '5x5 kernel(Test accuracy - {})'.format(s2[1])
    legend2 = '10x10 kernel(Test accuracy - {})'.format(s2[1])
    plt.legend([legend1, legend2], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_cnn_{}.png'.format(plot_type))
    plt.close()


def test_no_of_kernels(plot_type='no_of_kernels'):
    h1, s1 = build_model(no_of_kernels=20)
    h2, s2 = build_model(no_of_kernels=32)

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.title('Model accuracy based on no of kernels')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = '20 kernels(Test accuracy - {})'.format(s1[1])
    legend2 = '32 kernels(Test accuracy - {})'.format(s2[1])
    plt.legend([legend1, legend2], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_cnn_{}.png'.format(plot_type))
    plt.close()


def test_overfitting(plot_type='overfitting'):
    h1, s1 = build_model(epochs=20)
    plt.plot(h1.history['acc'])
    plt.plot(h1.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_cnn_test_train.png'.format(plot_type))
    plt.close()

    plt.plot(h1.history['loss'])
    plt.plot(h1.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(PLOTS_DIR + '/loss_cnn_test_train.png'.format(plot_type))
    plt.close()

    h2, s2 = build_model(epochs=20, add_dropout=True)
    h3, s3 = build_model(epochs=20, l2_reg=0.01)
    plt.plot(h1.history['val_loss'])
    plt.plot(h2.history['val_loss'])
    plt.plot(h3.history['val_loss'])
    plt.title('Model loss using different overfitting techniques')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = 'No technique used(Test accuracy - {})'.format(s1[1])
    legend2 = 'Dropout(Test accuracy - {})'.format(s2[1])
    legend3 = 'L2 regularization(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='upper right')
    plt.savefig(PLOTS_DIR + '/loss_cnn_{}.png'.format(plot_type))
    plt.close()


if __name__ == '__main__':
    # test_no_of_layers()
    # test_activation_fun()
    # test_kernel_size()
    # test_no_of_kernels()
    # test_overfitting()
    pass
