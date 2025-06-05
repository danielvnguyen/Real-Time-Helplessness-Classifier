import torch
import numpy as np
from dataset import HelplessnessVideoDataset
from torchvision import transforms
from tqdm import tqdm

# this script is inspired by this sample code: https://gist.github.com/Huud/8e0823fa7be2dcd1bb9f3c418cb94c19

# Load the full dataset to calculate the mean and std of our collected samples
full_dataset = HelplessnessVideoDataset(
    root_dir='../processed_frames',
    transform=transforms.ToTensor()
)

# Create variables to store running mean and std
mean = np.array([0.0, 0.0, 0.0])
std = np.array([0.0, 0.0, 0.0])
num_samples = len(full_dataset)
num_frames = 90
num_channels = 3

# Calculate the mean in the first run
for i in tqdm(range(num_samples)):
    frames = full_dataset[i][0].numpy()
    for j in range(num_channels):
        mean[j] += np.mean(frames[j, :, :, :]) # take the mean across all frames in the sequence

mean = (mean / num_samples)

# Calculate the std in the second run using the mean of the dataset
for i in tqdm(range(num_samples)):
    frames = full_dataset[i][0].numpy()
    for j in range(num_channels):
        std[j] += ((frames[j, :, :, :] - mean[j])**2).sum() / (frames.shape[1] * frames.shape[2] * frames.shape[3])

std = np.sqrt(std / num_samples)

# Print the final mean and std of the dataset
print(f'Mean: {mean}')
print(f'Standard deviation: {std}')
