import pandas as pd
import csv
import torch
import clip
from PIL import Image

# Read the CSV file into a pandas DataFrame
csv_file_path = '16_lables.csv'  # corrected the file name
data = pd.read_csv(csv_file_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

label_groups = [
    ['a photo of spring', 'a photo of summer', 'a photo of autumn', 'a photo of winter'],
    ['blooming flowers', 'dense green trees', 'colorful leaves', 'bare trees'],
    ['a colorful wildflowers', 'sunlit wheat fields', 'yellow leaves', 'icy rivers'],
    ['green grass and fresh buds on trees', 'dense green trees in summer', 'brown leaves in fall', 'cold weather with bare trees']
]

# Function to get prediction scores
def get_prediction_scores(image, labels, model, preprocess, device):
    image_input = preprocess(Image.open(image)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=1)
        return probs.cpu().numpy()

# Open a CSV file for each label group outside the loop to avoid overwriting
csv_files = []
csv_writers = []
for idx, label_group in enumerate(label_groups):
    csvfile = open(f'all_result_label_group_{idx + 1}.csv', 'w', newline='')
    fieldnames = ['Image', 'Prediction_for_spring', 'Prediction_for_summer', 'Prediction_for_autumn', 'Prediction_for_winter']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csv_files.append(csvfile)
    csv_writers.append(writer)

# Iterate over images and label groups to get predictions and write to files
for image in data['Image']:
    for idx, label_group in enumerate(label_groups):
        prediction_results = get_prediction_scores(image, label_group, model, preprocess, device)
        print(prediction_results)
        # Find the label with the highest probability (prediction) and its score (confidence)
        # Write the prediction result to the corresponding CSV file
        csv_writers[idx].writerow({'Image': image, 'Prediction_for_spring': prediction_results[0][0], 
                                   'Prediction_for_summer': prediction_results[0][1], 'Prediction_for_autumn': prediction_results[0][2], 'Prediction_for_winter': prediction_results[0][3]})

# Close all CSV files
for csvfile in csv_files:
    csvfile.close()






