import preprocessing
import models

X_train,y_train,X_test,y_test = get_data()
model_ANN=create_ANN()
model_CNN=create_CNN()
model_ANN.summary()
model_CNN.summary()

history_ANN = model_ANN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)

history_CNN = model_CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3)



