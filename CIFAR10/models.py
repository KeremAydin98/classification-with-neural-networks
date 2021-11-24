from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential

def create_CNN(X_train):

  model = Sequential()

  model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=X_train[0].shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=X_train[0].shape, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dense(256,activation='relu'))

  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


  return model
