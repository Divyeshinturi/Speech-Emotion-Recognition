import kagglehub
import os

path = kagglehub.dataset_download("ejlok1/cremad")

print("\nDataset Location:")
print(path)

print("\nContents:")
print(os.listdir(path))