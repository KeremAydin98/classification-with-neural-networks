from preprocessing import data_preprocessing
from models import create_CNN
from evaluate import evaluate_model, compare_pred_with_actual
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = data_preprocessing()

classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"}

model = create_CNN(X_train)

early_stop = EarlyStopping(monitor='val_loss', mode='min',patience=5)

model_history = model.fit(X_train, y_train, validation_data=(X_test,y_test),callbacks=[early_stop],epochs=15)

evaluate_model(model,X_test,y_test)

compare_pred_with_actual(model,X_test,y_test,classes)

plt.figure(1)
plt.plot(model_history.history['loss'],label='loss')
plt.plot(model_history.history['val_loss'],label='val_loss')
plt.show()

plt.figure(2)
plt.plot(model_history.history['accuracy'],label='accuracy')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()