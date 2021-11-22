from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential

def create_ANN():

  model=Sequential()

  model.add(Flatten(input_shape=(28,28)))
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(128,activation='relu'))


  model.add(Dense(10,activation='softmax'))

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

  return model


def create_CNN():

  model=Sequential()

  model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.1))

  model.add(Conv2D(64,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.1))

  model.add(Conv2D(128,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.1))

  model.add(Flatten())

  model.add(Dense(128,activation='relu'))
  model.add(Dense(10,activation='softmax'))

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

  return model