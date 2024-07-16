# Modifying the script to change the last 16 headers
import pandas as pd
# New headers for the last 16 columns
new_headers = [
    'a photo of spring', 'a photo of summer', 'a photo of autumn', 'a photo of winter',
    'blooming flowers', 'dense green trees', 'colorful leaves', 'bare trees',
    'a colorful wildflowers', 'sunlit wheat fields', 'yellow leaves', 'icy rivers',
    'green grass and fresh buds on trees', 'dense green trees in summer', 'brown leaves in fall', 'cold weather with bare trees'
]

# Adjusting the script

csv_files = ['16_labels_predictions.csv', 'all_result_label_group_1.csv', 'all_result_label_group_2.csv', 'all_result_label_group_3.csv', 'all_result_label_group_4.csv']

combined_df = pd.read_csv(csv_files[0], usecols=[0, 1])

for file in csv_files[1:]:
    next_df = pd.read_csv(file)
    # Drop the 'Image' column from next_df to avoid duplication except for the first
    next_df = next_df.drop('Image', axis=1)

    # Concatenate along the columns
    combined_df = pd.concat([combined_df, next_df], axis=1)

# Changing the last 16 headers
combined_df.columns = list(combined_df.columns[:-16]) + new_headers

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_csv_with_all_labels_seasons.csv', index=False)
