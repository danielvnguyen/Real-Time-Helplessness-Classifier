import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

############################################
# Create CSV sheet for Ratings
############################################
# # folder names
# folders = ["extreme-helpless", "little_helplessness", "no-helpless"]
#
# # Collect all video file paths
# file_list = []
#
# for folder in folders:
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                 relative_path = os.path.join(folder, file)
#                 file_list.append(relative_path)
#
# # Sort alphabetically
# file_list.sort()
#
# # Create a DataFrame
# df = pd.DataFrame({
#     "Video Filename": file_list,
#     "Rater 1": "",
#     "Rater 2": "",
#     "Rater 3": ""
# })
#
# # Save to CSV
# df.to_csv("cohen_kappa_rating_sheet.csv", index=False)
#
# print("CSV file created successfully")

############################################
# Obtain Cohen's Kappa Score from CSV
############################################

# Load the CSV
df = pd.read_csv("cohen_kappa_rating_sheet.csv")

# Obtain annotations for each rater
annotations_1 = df["Rater 1"]
annotations_2 = df["Rater 2"]
annotations_3 = df["Rater 3"]

# Compute Cohen's Kappa Scores for each combination
kappa_1_2 = cohen_kappa_score(annotations_1, annotations_2)
kappa_1_3 = cohen_kappa_score(annotations_1, annotations_3)
kappa_2_3 = cohen_kappa_score(annotations_2, annotations_3)

print("Cohen's Kappa Scores:")
print(f"Rater 1 vs. 2: {kappa_1_2:.2f}")
print(f"Rater 1 vs. 3: {kappa_1_3:.2f}")
print(f"Rater 2 vs. 3: {kappa_2_3:.2f}")

# Compute average for Inter-Rater Agreement Score
avg = (kappa_1_2 + kappa_1_3 + kappa_2_3) / 3
print(f"Average Score: {avg:.2f}")