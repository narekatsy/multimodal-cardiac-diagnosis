import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """
    Resample an image to a new spacing using SimpleITK.
    
    Args:
        image (SimpleITK.Image): Input image.
        new_spacing (list): Desired voxel spacing (e.g., [1.0, 1.0, 1.0]).
    
    Returns:
        SimpleITK.Image: Resampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(original_size[0] * original_spacing[0] / new_spacing[0]),
        int(original_size[1] * original_spacing[1] / new_spacing[1]),
        int(original_size[2] * original_spacing[2] / new_spacing[2]),
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)
    
    return resampled_image

def center_crop(img_data, target_shape=(256, 256, 10)):
    """
    Crop the center of an image to a target shape.
    
    Args:
        img_data (numpy.ndarray): Input image data.
        target_shape (tuple): Desired shape (x, y, z).
    
    Returns:
        numpy.ndarray: Cropped image data.
    """
    start_x = (img_data.shape[0] - target_shape[0]) // 2
    start_y = (img_data.shape[1] - target_shape[1]) // 2
    start_z = (img_data.shape[2] - target_shape[2]) // 2
    
    cropped = img_data[
        start_x:start_x + target_shape[0],
        start_y:start_y + target_shape[1],
        start_z:start_z + target_shape[2],
    ]
    return cropped

def preprocess_mri(nii_path, target_shape=(256, 256, 10)):
    """
    Preprocess an MRI scan (resample, normalize, crop).
    
    Args:
        nii_path (str): Path to the .nii.gz file.
        target_shape (tuple): Desired output shape (x, y, z).
    
    Returns:
        numpy.ndarray: Preprocessed image data.
    """
    img = nib.load(nii_path)
    img = nib.as_closest_canonical(img)
    img_data = img.get_fdata()

    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    sitk_img = sitk.ReadImage(nii_path)
    resampled_img = resample_image(sitk_img, new_spacing=[1.0, 1.0, 1.0])
    resampled_data = sitk.GetArrayFromImage(resampled_img)

    cropped_data = center_crop(resampled_data, target_shape)

    return cropped_data

if __name__ == "__main__":
    nii_path = os.path.join("data", "MRI", "training", "patient001", "patient001_4d.nii.gz")

    preprocessed_data = preprocess_mri(nii_path)
    
    plt.imshow(preprocessed_data[:, :, 5, 0], cmap="gray")
    plt.title("Preprocessed MRI Slice")
    plt.axis("off")
    plt.show()

