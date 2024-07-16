import pandas as pd
import csv
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
csv_file_path = '16_lables.csv'
data = pd.read_csv(csv_file_path)
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

# Convert all score columns to numeric, handling non-numeric values by converting them to NaN
for col in data.columns[2:]:  # Convert columns starting from the third column
    data[col] = pd.to_numeric(data[col], errors='coerce')

# After conversion, fill NaN values with zero
data.iloc[:, 2:] = data.iloc[:, 2:].fillna(0)

# Explicitly check for any remaining non-numeric columns
for col in data.columns[2:]:
    if data[col].dtype == 'object':
        print(f"Column {col} contains non-numeric data after conversion.")

# Assuming all columns are now numeric, calculate the 'Confidence' score
try:
    data['Confidence'] = data.iloc[:, 2:].max(axis=1)
except TypeError as e:
    print(f"An error occurred: {e}")

# Calculate 'Predicted Season' based on the highest score
# Note: The calculation of 'Predicted Season' depends on the correct ordering of columns
# and the assumption that every four columns represent the scores for one season.
data['Predicted Season'] = data.iloc[:, 2:].apply(
    lambda row: seasons[int(row.argmax() / 4)], axis=1
)



# # Print the first 50 rows to verify the output
# print(data[['Image', 'Ground Truth Season', 'Predicted Season', 'Confidence']].head(50))

# # Select only the required columns to save to a new CSV
# selected_columns = data[['Image', 'Ground Truth Season', 'Predicted Season', 'Confidence']]

# # Save the selected columns to a new CSV file
# selected_columns.to_csv('16_labels_prediction_results.csv', index=False)


with open('16_labels_predictions.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header
    csvwriter.writerow(['Image', 'Ground Truth Season', 'Predicted Season', 'Confidence'])
    
    # Write the data rows
    for index, row in data.iterrows():
        csvwriter.writerow([row['Image'], row['Ground Truth Season'], row['Predicted Season'], row['Confidence']])



# Calculate confusion matrix
conf_matrix = confusion_matrix(data['Ground Truth Season'], data['Predicted Season'], labels=seasons)

# Calculate classification report
class_report = classification_report(data['Ground Truth Season'], data['Predicted Season'], labels=seasons)

# Print the classification report
print(class_report)

# Calculate accuracy
accuracy = (data['Ground Truth Season'] == data['Predicted Season']).mean()
print(f'Accuracy: {accuracy:.2f}')

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=seasons, yticklabels=seasons)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()