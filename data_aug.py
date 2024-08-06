import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """Load image and mask file paths for training and testing data."""
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.jpg")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.jpg")))

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    """Augment data by applying transformations and save the augmented data."""
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extract the name from the image path
        name = os.path.basename(x).split(".")[0]

        # Read image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.imread(y)

        if augment:
            # Apply augmentations
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        # Save augmented images and masks
        for index, (i, m) in enumerate(zip(X, Y)):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the data
    data_path = "dataset"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    # Create directories to save the augmented data
    create_dir(os.path.join("new_data", "train", "image"))
    create_dir(os.path.join("new_data", "train", "mask"))
    create_dir(os.path.join("new_data", "test", "image"))
    create_dir(os.path.join("new_data", "test", "mask"))

    # Perform data augmentation
    augment_data(train_x, train_y, os.path.join("new_data", "train"), augment=True)
    augment_data(test_x, test_y, os.path.join("new_data", "test"), augment=False)
