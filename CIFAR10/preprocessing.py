from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def data_preprocessing():
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  X_train = X_train / X_train.max()
  X_test = X_test / X_test.max()

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)


  return X_train, y_train, X_test, y_test