import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

input_dir = "split_data/MRI/training/"
output_dir = "processed_data/MRI/training/"
os.makedirs(output_dir, exist_ok=True)

TARGET_SHAPE = (128, 128, 16)

def load_mri(file_path):
    """ Load a .nii.gz MRI file and return a numpy array. """
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def normalize_mri(image):
    """ Normalize MRI intensity values to range [0, 1]. """
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image

def preprocess_mri(file_path):
    """ Load, normalize, resize, and save MRI. """
    filename = os.path.basename(file_path).replace(".nii.gz", ".npy")
    image = load_mri(file_path)

    print(f"Original shape: {image.shape}")

    if len(image.shape) == 4:
        image = image[..., 0]

    image = normalize_mri(image)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    image_resized = F.interpolate(image_tensor, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
    image_resized = image_resized.squeeze().numpy()

    np.save(os.path.join(output_dir, filename), image_resized)

    return image_resized

def process_all_mris():
    """ Process all MRI scans in dataset. """
    mri_files = glob(os.path.join(input_dir, "*/*.nii.gz"))
    
    for file in mri_files:
        print(f"Processing {file}...")
        preprocess_mri(file)

    print("MRI preprocessing completed!")

if __name__ == "__main__":
    process_all_mris()
