import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
csv_file_path_1 = '16_lables.csv'
data = pd.read_csv(csv_file_path_1)


csv_file_path_2 = 'result_label_group_4.csv'
prediction = pd.read_csv(csv_file_path_2)

# Calculate confusion matrix
conf_matrix = confusion_matrix(data['Ground Truth Season'], prediction['Prediction'], labels=seasons)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Calculate classification report
class_report = classification_report(data['Ground Truth Season'], prediction['Prediction'], labels=seasons)

# Print the classification report
print(class_report)

sns.set(font_scale=2)

# Plot the normalized confusion matrix
plt.figure(figsize=(10, 8))
ax = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap='Blues', xticklabels=seasons, yticklabels=seasons)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.tick_params(axis='both', which='major', labelsize=20)
plt.show()