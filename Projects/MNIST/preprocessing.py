from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalization between 0-1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train / X_train.max()
    X_test = X_test / X_test.max()

    # Transforming labels into categorical data using one-hot-encoding

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train,y_train,X_test,y_test





