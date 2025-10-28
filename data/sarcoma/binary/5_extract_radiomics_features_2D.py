import os
import torch
from radiomics import featureextractor
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import numpy as np

def extract_features_2d(dataset_path):
    """
    Extracts radiomics features from 2D PNG images and their corresponding masks.
    Saves the features for each image in a separate .pt file.

    Args:
        dataset_path (str): Path containing PNG image/mask pairs.
                            Assumes images are named with '_image.png' and masks with '_mask.png'.
    """

    # ---------------------------------------------------------------------
    # File collection
    # ---------------------------------------------------------------------
    image_paths = sorted([f for f in glob(os.path.join(dataset_path, "*axial.png"))])
    image_paths = sorted([f for f in image_paths if not "label" in f])
    mask_paths = sorted([f for f in glob(os.path.join(dataset_path, "*label*.png"))])

    assert len(image_paths) == len(mask_paths), "Image and mask lists must have same length."

    # ---------------------------------------------------------------------
    # PyRadiomics setup for 2D images
    # ---------------------------------------------------------------------
    settings = {
        "force2D": True,             # Tell PyRadiomics to handle 2D inputs
        "force2Ddimension": 0,       # Use the slice in the first dimension
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "enableCExtensions": True,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    print("Enabled feature classes:")
    for cls, features in extractor.enabledFeatures.items():
        print(f"  {cls}: {features}")

    # ---------------------------------------------------------------------
    # Feature extraction loop
    # ---------------------------------------------------------------------
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):

        try:
            # Read 2D images as SimpleITK images
            image = sitk.ReadImage(img_path, sitk.sitkFloat32)
            mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
            mask = mask / np.max(sitk.GetArrayFromImage(mask)) 

            # Optional: sanity check for same dimensions
            if image.GetSize() != mask.GetSize():
                raise ValueError(f"Image and mask size mismatch for {img_path}")

            # Extract features
            result = extractor.execute(image, mask)

            # Keep only non-diagnostic entries
            features_dict = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}

            # Save to .pt file
            save_path = img_path.replace("_preprocessed_axial.png", "_features2D.pt")
            torch.save(features_dict, save_path)

        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

if __name__ == "__main__":

    for dataset_path in ["/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/data/sarcoma/binary/T1FsGd",
                         "/home/johannes/Data/SSD_2.0TB/ICL-VL/johannes/data/sarcoma/binary/T2Fs"]:
        extract_features_2d(dataset_path)
