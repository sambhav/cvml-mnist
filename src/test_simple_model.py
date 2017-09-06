import os
from os.path import dirname, realpath

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2, l1
import matplotlib.pyplot as plt

PROJECT_DIR = dirname(dirname(realpath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


def build_model(batch_size=200, num_classes=10, epochs=20, layers=2, activation_fn='relu', add_dropout=False, l2_reg = 0.0):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(100, kernel_regularizer=l2(l2_reg), activation=activation_fn, input_shape=(784,)))
    if add_dropout:
        model.add(Dropout(0.2))

    if layers >= 2:
        model.add(Dense(50, kernel_regularizer=l2(l2_reg), activation=activation_fn))
        if add_dropout:
            model.add(Dropout(0.2))
    if layers >= 3:
        model.add(Dense(30, kernel_regularizer=l2(l2_reg), activation=activation_fn))
        if add_dropout:
            model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
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


def test_no_of_layers(plot_type = 'layers'):
    h1,s1 = build_model(layers=1)
    h2,s2 = build_model(layers=2)
    h3,s3 = build_model(layers=3)

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.plot(h3.history['val_acc'])
    plt.title('Model accuracy based on no. of hidden layers')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = '1 layer(Test accuracy - {})'.format(s1[1])
    legend2 = '2 layers(Test accuracy - {})'.format(s2[1])
    legend3 = '3 layers(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_simple_{}.png'.format(plot_type))
    plt.close()


def test_epochs(plot_type = 'epochs'):
    h1,s1 = build_model(epochs=10)
    h2,s2 = build_model(epochs=20)
    h3,s3 = build_model(epochs=50)

    plt.plot(h1.history['val_acc'])
    plt.plot(h2.history['val_acc'])
    plt.plot(h3.history['val_acc'])
    plt.title('Model accuracy based on no. of epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    legend1 = '10 epochs(Test accuracy - {})'.format(s1[1])
    legend2 = '20 epochs(Test accuracy - {})'.format(s2[1])
    legend3 = '50 epochs(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_simple_{}.png'.format(plot_type))
    plt.close()


def test_activation_fun(plot_type = 'activation'):
    h1,s1 = build_model(activation_fn='relu')
    h2,s2 = build_model(activation_fn='sigmoid')
    h3,s3 = build_model(activation_fn='tanh')

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
    plt.savefig(PLOTS_DIR + '/acc_simple_{}.png'.format(plot_type))
    plt.close()

def test_overfitting(plot_type = 'overfitting'):
    h1,s1 = build_model(epochs=40)
    plt.plot(h1.history['acc'])
    plt.plot(h1.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(PLOTS_DIR + '/acc_simple_test_train.png'.format(plot_type))
    plt.close()

    plt.plot(h1.history['loss'])
    plt.plot(h1.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(PLOTS_DIR + '/loss_simple_test_train.png'.format(plot_type))
    plt.close()

    h2,s2 = build_model(epochs=40, add_dropout = True)
    h3,s3 = build_model(epochs=40, l2_reg = 0.0001)
    plt.plot(h1.history['val_loss'])
    plt.plot(h2.history['val_loss'])
    plt.plot(h3.history['val_loss'])
    plt.title('Model accuracy using different overfitting techniques')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    legend1 = 'No technique used(Test accuracy - {})'.format(s1[1])
    legend2 = 'Dropout(Test accuracy - {})'.format(s2[1])
    legend3 = 'L2 regularization(Test accuracy - {})'.format(s3[1])
    plt.legend([legend1, legend2, legend3], loc='lower right')
    plt.savefig(PLOTS_DIR + '/loss_simple_{}.png'.format(plot_type))
    plt.close()


if __name__ == '__main__':
    # test_no_of_layers()
    # # test_epochs()
    # test_activation_fun()
    # test_overfitting()
