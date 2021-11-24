from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(model,X_test,y_test):

    print(str(model))
    print("\n")

    y_pred =model.predict(X_test)
    y_pred =np.argmax(y_pred, axis=1)
    y_test_class =np.argmax(y_test, axis=1)

    print("Classification Report: \n")
    print(classification_report(y_test_class ,y_pred))

    data ={"y_Actual" :y_test_class ,"y_Predicted" :y_pred}
    df = pd.DataFrame(data, columns=['y_Actual' ,'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

    plt.figure(figsize=(10 ,10))
    sns.heatmap(confusion_matrix ,annot=True)
    plt.show()
    print("\n")


def compare_pred_with_actual(model,X_test,y_test,classes):
    figure, ax = plt.subplots(1, 5, figsize=(10, 10))

    for i in range(5):
        random_value = np.random.randint(low=0, high=10000)
        y_pred = model.predict(X_test)

        y_predicted = np.argmax(y_pred[random_value], axis=0)
        y_original = np.argmax(y_test[random_value], axis=0)

        predict_class = classes[y_predicted]
        original_class = classes[y_original]

        ax[i].imshow(X_test[random_value], cmap='gray')
        ax[i].set_title("True: %s \nPredict: %s" % (str(original_class), str(predict_class)))