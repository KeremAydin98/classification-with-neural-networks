from preprocessing import get_data
from models import create_ANN,create_CNN
from evaluate import evaluate_model,compare_pred_with_actual

#First ANN
X_train,y_train,X_test,y_test = get_data()
model_ANN=create_ANN()

history_ANN = model_ANN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)

evaluate_model(model_ANN,X_test,y_test)

compare_pred_with_actual(model_ANN,X_test,y_test)

#Then CNN
X_train,y_train,X_test,y_test = get_data()
model_CNN=create_CNN()
history_CNN = model_CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3)

evaluate_model(model_CNN,X_test,y_test)
compare_pred_with_actual(model_CNN,X_test,y_test)

plot1 = plt.figure(1)
plt.plot(history_ANN.history['accuracy'],label='accuracy')
plt.title("ANN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plot2 = plt.figure(2)

plt.plot(history_CNN.history['accuracy'],label='accuracy')
plt.title("CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()