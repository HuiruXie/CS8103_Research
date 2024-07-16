from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns


csv_files = 'combined_csv_with_all_labels_seasons.csv'

data = pd.read_csv(csv_files)

# Encoding the target variable
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['Ground Truth Season'])

# One-hot encoding for the neural network output
encoded_labels = to_categorical(encoded_labels)

# Splitting the dataset into features and target
features = data.iloc[:, 2:]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Building the neural network model
model = Sequential()
model.add(Dense(64, input_dim=16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(encoded_labels.shape[1], activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs= 10, batch_size=10, verbose=0)

# Plotting the learning curve, loss curve, and accuracy
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Evaluating the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Predictions for confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
conf_matrix_report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)

# Normalizing the confusion matrix
cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Function to plot the normalized confusion matrix
def plot_confusion_matrix(cm, classes, title='Normalized Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(14, 10)) # Increased figure size for better visibility
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=35) # Increased title font size
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=20)  # Horizontal x-ticks
    plt.yticks(tick_marks, classes, rotation=90, fontsize=25)  # Vertical y-ticks

    # Loop over data dimensions and create text annotations.
    fmt = '.2%' # Correct format for percentage
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=30) # Increased font size for text annotations
    plt.tight_layout()
    plt.ylabel('True label', fontsize=35)
    plt.xlabel('Predicted label', fontsize=35)
    plt.show()

# Plotting the normalized confusion matrix

plot_confusion_matrix(cm_normalized, classes=label_encoder.classes_)

# Displaying the accuracy
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Displaying the classification report
print("Classification Report:")
print(conf_matrix_report)


# 打印模型的架构
model.summary()

# 打印训练过程中的准确率和损失值
history_dict = history.history
print("Training and validation history: ")
print(history_dict)

# 打印最终在测试集上的评估结果
print(f"Final test loss: {test_loss}")
print(f"Final test accuracy: {test_accuracy * 100:.2f}%")

