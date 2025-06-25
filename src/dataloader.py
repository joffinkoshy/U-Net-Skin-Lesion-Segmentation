import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import traceback
import tensorflow as tf # Import TensorFlow for augmentation operations

# This is for the 2D Dermatology project (ISIC 2016)

def augment_image_mask(image, mask):
    """
    Applies random geometric and photometric augmentations to an image and its mask.
    Ensure that geometric transformations are applied identically to both image and mask.
    """
    # Convert to TensorFlow tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    # Add a channel dimension to mask if it's 2D (e.g., (128,128) to (128,128,1))
    if mask.shape.ndims == 2:
        mask = tf.expand_dims(mask, axis=-1)

    # 1. Random Flip Left-Right
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 2. Random Flip Up-Down
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # 3. Random Brightness (Image only)
    image = tf.image.random_brightness(image, max_delta=0.2) # Max delta for brightness adjustment

    # 4. Random Contrast (Image only)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) # Factor for contrast adjustment

    # Ensure mask remains binary after any potential float operations (though tf.image ops preserve this if input is binary)
    mask = tf.round(mask)

    return image.numpy(), mask.numpy() # Convert back to numpy arrays for consistency with current pipeline


def load_isic_data(data_dir, img_height=128, img_width=128,
                   apply_augmentation=False): # Keep the flag here
    """
    Loads ISIC 2016 images and masks, resizes them, and prepares for training.

    Args:
        data_dir (str): Path to the 'isic_2016' directory (e.g., '/Users/joffinkoshy/Desktop/MedicalSegmentation/data/isic_2016').
        img_height (int): Target height for resizing images/masks.
        img_width (int): Target width for resizing images/masks.
        apply_augmentation (bool): Whether to apply geometric data augmentation.

    Returns:
        tuple: (images, masks, image_ids) as numpy arrays and list.
    """
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')

    print(f"Checking if image directory exists: {image_dir}")
    if not os.path.exists(image_dir):
        print(f"CRITICAL ERROR: Image directory NOT FOUND at {data_dir}.")
        print("Please ensure the path is correct and the folder exists.")
        print("This is the most common cause of error. Check spelling and capitalization.")
        return None, None, None

    print(f"Checking if mask directory exists: {mask_dir}")
    if not os.path.exists(mask_dir):
        print(f"CRITICAL ERROR: Mask directory NOT FOUND at {mask_dir}")
        return None, None, None

    image_files = sorted(os.listdir(image_dir))
    image_files = [f for f in image_files if f.endswith('.jpg')]

    if not image_files:
        print(f"WARNING: No .jpg images found in {image_dir}. Please check content.")
        return None, None, None

    images = []
    masks = []
    loaded_image_ids = []

    print(f"Attempting to load {len(image_files)} image-mask pairs...")

    for i, img_filename in enumerate(image_files):
        img_path = os.path.join(image_dir, img_filename)
        mask_filename = img_filename.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        if i < 5 or i % 100 == 0:
            print(f"Processing ({i + 1}/{len(image_files)}): Image: {img_path}, Mask: {mask_path}")

        if not os.path.exists(mask_path):
            print(f"WARNING: Mask for {img_filename} not found at {mask_path}. Skipping.")
            continue

        try:
            img = imread(img_path)
            mask = imread(mask_path, as_gray=True)

            # print(
            #     f"DEBUG: After imread({img_filename}): Shape={img.shape}, Dtype={img.dtype}, Min={img.min()}, Max={img.max()}")

            if img.max() > 1.0:
                img_processed_for_scale = img.astype(np.float32) / 255.0
            else:
                img_processed_for_scale = img.astype(np.float32)

            if img_processed_for_scale.ndim == 2:
                img_processed_for_scale = np.stack(
                    [img_processed_for_scale, img_processed_for_scale, img_processed_for_scale], axis=-1)
            elif img_processed_for_scale.shape[-1] == 4:
                img_processed_for_scale = img_processed_for_scale[..., :3]

            img_resized = resize(img_processed_for_scale, (img_height, img_width, 3), anti_aliasing=True)
            mask_resized = resize(mask, (img_height, img_width), anti_aliasing=False)
            mask_resized = (mask_resized > 0.5).astype(np.float32)

            # --- Apply augmentation here if flag is True ---
            if apply_augmentation:
                img_resized, mask_resized = augment_image_mask(img_resized, mask_resized)


            # print(
            #     f"DEBUG: Before appending to images[]: Shape={img_resized.shape}, Dtype={img_resized.dtype}, Min={img_resized.min():.4f}, Max={img_resized.max():.4f}")

            images.append(img_resized)
            masks.append(mask_resized)
            loaded_image_ids.append(img_filename)

        except Exception as e:
            print(f"ERROR: Could not load or process {img_filename}: {e}")
            traceback.print_exc()
            continue

    if not images:  # Check if any images were successfully loaded
        print("No image-mask pairs were successfully loaded. Please check data files and paths.")
        return None, None, None

    images = np.array(images, dtype=np.float32)
    masks = np.expand_dims(np.array(masks, dtype=np.float32), axis=-1)

    print(f"\nSuccessfully loaded {len(images)} image-mask pairs.")
    return images, masks, loaded_image_ids


def split_data(images, masks, test_size=0.2, random_state=42):
    """
    Splits data into training and validation/test sets.
    """
    if images is None or masks is None:
        print("Cannot split data: images or masks are None.")
        return None, None, None, None, None, None

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def visualize_sample(image, mask, prediction=None, title="Sample", save_path=None):
    """
    Visualizes an image, its ground truth mask, and optionally a prediction.
    Can also save the figure to a specified path.
    """
    fig, axes = plt.subplots(1, 2 + (1 if prediction is not None else 0), figsize=(12, 6))

    axes[0].imshow(image, vmin=0, vmax=1)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(image, vmin=0, vmax=1)
    axes[1].imshow(np.squeeze(mask), alpha=0.5, cmap='magma')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    if prediction is not None:
        axes[2].imshow(image, vmin=0, vmax=1)
        axes[2].imshow(prediction[:, :, 0] if prediction.ndim == 3 else prediction, alpha=0.5, cmap='magma')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"Prediction visualization saved to: {save_path}")

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # Make sure to adjust this path if you're running dataloader.py directly for testing
    # If your project root is 'MedicalSegmentation' and data is in 'MedicalSegmentation/data/isic_2016'
    # Then for dataloader.py run, it should be:
    # DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'isic_2016')
    DATA_DIR = "/Users/joffinkoshy/Desktop/MedicalSegmentation/data/isic_2016" # Original path from user

    print(f"--- Starting Data Loading Process ---")
    print(f"Attempting to load data from: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Primary data directory NOT FOUND at {DATA_DIR}.")
        print("Please ensure the path is correct and the folder exists.")
        print("This is the most common cause of error. Check spelling and capitalization.")
    else:
        # Test with augmentation enabled
        images_aug, masks_aug, image_ids_aug = load_isic_data(DATA_DIR, img_height=128, img_width=128,
                                                          apply_augmentation=True)

        if images_aug is not None and masks_aug is not None:
            print(f"Loaded (Augmented) images shape: {images_aug.shape}")
            print(f"Loaded (Augmented) masks shape: {masks_aug.shape}")

            print("\nVisualizing a few augmented samples:")
            num_samples_to_visualize = min(5, len(images_aug))
            for i in range(num_samples_to_visualize):
                idx = np.random.randint(0, len(images_aug))
                visualize_sample(images_aug[idx], masks_aug[idx], title=f"Augmented Sample {idx}")

            X_train, y_train, X_val, y_val, X_test, y_test = split_data(images_aug, masks_aug)
            if X_train is not None:
                print(f"\nTrain set size: {len(X_train)}")
                print(f"Validation set size: {len(X_val)}")
                print(f"Test set size: {len(X_test)}")
        else:
            print("\nData loading failed. No data to process further.")

    print(f"--- Data Loading Process Finished ---")