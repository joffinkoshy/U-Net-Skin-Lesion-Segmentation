"""
U-Net Skin Lesion Segmentation Training Script
Supports both standard U-Net and enhanced Attention U-Net architectures
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import custom modules
from dataloader import load_isic_data, visualize_sample
from unet_model import unet_model
from models.attention_unet import attention_unet_model
from data.advanced_augmentation import advanced_augmentation, preprocess_image
from utils.training_utils import (get_optimizer, combined_loss, dice_coef, hausdorff_distance,
                                get_callbacks, enable_mixed_precision, monte_carlo_dropout_prediction)
from utils.post_processing import post_process_prediction
from visualization.visualization_utils import (visualize_segmentation_comparison, generate_grad_cam,
                                             visualize_grad_cam, plot_training_history)

# Enable mixed precision training
enable_mixed_precision()

# --- Custom Metrics/Losses for Standard U-Net ---
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def load_and_preprocess_data(data_dir, img_height=128, img_width=128, apply_augmentation=True, enhanced_mode=False):
    """
    Load and preprocess ISIC 2016 dataset.

    Args:
        data_dir: Path to data directory
        img_height: Target height
        img_width: Target width
        apply_augmentation: Whether to apply augmentation
        enhanced_mode: Whether to use enhanced preprocessing

    Returns:
        Tuple of (images, masks, image_ids)
    """
    # Load data using original function
    images, masks, image_ids = load_isic_data(data_dir, img_height, img_width, apply_augmentation=False)

    if images is None or masks is None:
        return None, None, None

    if not enhanced_mode:
        return images, masks, image_ids

    # Apply advanced preprocessing and augmentation for enhanced mode
    processed_images = []
    processed_masks = []

    print(f"Applying advanced preprocessing to {len(images)} images...")

    for i, (img, mask) in enumerate(zip(images, masks)):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(images)}...")

        # Preprocess image
        img = preprocess_image(img.squeeze())
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)

        # Apply advanced augmentation if enabled
        if apply_augmentation:
            img, mask = advanced_augmentation(img, mask.squeeze())
            mask = np.expand_dims(mask, axis=-1)
        else:
            mask = mask.squeeze()
            mask = (mask > 0.5).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

        processed_images.append(img)
        processed_masks.append(mask)

    return np.array(processed_images, dtype=np.float32), np.array(processed_masks, dtype=np.float32), image_ids

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='U-Net Skin Lesion Segmentation Training')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'enhanced'],
                       help='Training mode: standard U-Net or enhanced Attention U-Net')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for optimizer')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    args = parser.parse_args()

    # Configuration
    CONFIG = {
        'MODE': args.mode,
        'IMAGE_HEIGHT': 128,
        'IMAGE_WIDTH': 128,
        'NUM_CHANNELS': 3,
        'INPUT_SHAPE': (128, 128, 3),
        'BATCH_SIZE': args.batch_size,
        'EPOCHS': args.epochs,
        'LEARNING_RATE': args.learning_rate,
        'NUM_FOLDS': 5,
        'OPTIMIZER': 'adamw' if args.mode == 'enhanced' else 'adam',
        'DROPOUT_RATE': 0.3 if args.mode == 'enhanced' else 0.0,
        'APPLY_AUGMENTATION': not args.no_augmentation,
        'USE_ADVANCED_LOSS': args.mode == 'enhanced',
        'ENABLE_UNCERTAINTY': args.mode == 'enhanced',
        'SAVE_RESULTS': True,
        'DEBUG': args.debug
    }

    if CONFIG['DEBUG']:
        print("=== DEBUG MODE ENABLED ===")
        print(f"Configuration: {CONFIG}")

    # Create results directory
    results_dir = os.path.join(project_root, 'Results_Enhanced' if CONFIG['MODE'] == 'enhanced' else 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Create models directory
    models_dir = os.path.join(project_root, 'models_enhanced' if CONFIG['MODE'] == 'enhanced' else 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Data directory
    data_dir = os.path.join(project_root, 'data', 'isic_2016')

    print(f"\n=== U-Net Skin Lesion Segmentation Training ({CONFIG['MODE'].upper()} MODE) ===")
    print(f"Results will be saved to: {results_dir}")

    # Load and preprocess data
    print("\n--- Loading and Preprocessing Data ---")
    images, masks, image_ids = load_and_preprocess_data(
        data_dir,
        img_height=CONFIG['IMAGE_HEIGHT'],
        img_width=CONFIG['IMAGE_WIDTH'],
        apply_augmentation=CONFIG['APPLY_AUGMENTATION'],
        enhanced_mode=CONFIG['MODE'] == 'enhanced'
    )

    if images is None or masks is None:
        print("Data loading failed. Exiting.")
        return

    print(f"Loaded {len(images)} images with shape: {images.shape}")
    print(f"Loaded {len(masks)} masks with shape: {masks.shape}")

    # Split data: 80% for training/validation, 20% for testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    print(f"Train/Val set: {len(X_train_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=CONFIG['NUM_FOLDS'], shuffle=True, random_state=42)

    fold_results = []
    histories = []
    best_models_paths = []

    print(f"\n--- Starting {CONFIG['NUM_FOLDS']}-Fold Cross-Validation ---")

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
        print(f"\n----- Fold {fold_num + 1}/{CONFIG['NUM_FOLDS']} -----")

        X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index]
        y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index]

        print(f"Fold {fold_num + 1} data: Train ({len(X_train_fold)}), Validation ({len(X_val_fold)})")

        # Clear previous session
        tf.keras.backend.clear_session()

        # Create model based on mode
        print(f"Building {'Attention U-Net' if CONFIG['MODE'] == 'enhanced' else 'Standard U-Net'} model...")

        if CONFIG['MODE'] == 'enhanced':
            model = attention_unet_model(
                input_size=CONFIG['INPUT_SHAPE'],
                dropout_rate=CONFIG['DROPOUT_RATE']
            )
        else:
            model = unet_model(input_size=CONFIG['INPUT_SHAPE'])

        # Compile model
        optimizer = get_optimizer(CONFIG['OPTIMIZER'], CONFIG['LEARNING_RATE'])

        if CONFIG['USE_ADVANCED_LOSS']:
            loss_function = combined_loss(bce_weight=0.4, dice_weight=0.5, focal_weight=0.1)
        else:
            loss_function = bce_dice_loss

        metrics = [
            tf.keras.metrics.MeanIoU(num_classes=2, name='mean_io_u'),
            dice_coef,
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]

        if CONFIG['MODE'] == 'enhanced':
            metrics.append(hausdorff_distance)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        print("Model compiled successfully.")

        # Callbacks
        fold_model_dir = os.path.join(models_dir, f'fold_{fold_num + 1}')
        os.makedirs(fold_model_dir, exist_ok=True)

        if CONFIG['MODE'] == 'enhanced':
            fold_model_path = os.path.join(fold_model_dir, f'attention_unet_fold_{fold_num + 1}.keras')
        else:
            fold_model_path = os.path.join(fold_model_dir, f'unet_fold_{fold_num + 1}.keras')

        best_models_paths.append(fold_model_path)

        callbacks = get_callbacks(
            fold_model_path,
            patience=20 if CONFIG['MODE'] == 'enhanced' else 10,
            reduce_lr_patience=10 if CONFIG['MODE'] == 'enhanced' else 5
        )

        # Train model
        print(f"Starting training for Fold {fold_num + 1}...")
        history = model.fit(
            X_train_fold, y_train_fold,
            batch_size=CONFIG['BATCH_SIZE'],
            epochs=CONFIG['EPOCHS'],
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=1
        )
        print(f"Training completed for Fold {fold_num + 1}.")

        histories.append(history)

        # Evaluate on test set
        print(f"Evaluating Fold {fold_num + 1} on test set...")

        # Load best model from this fold
        custom_objects = {
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'bce_dice_loss': bce_dice_loss
        }

        if CONFIG['MODE'] == 'enhanced':
            custom_objects['hausdorff_distance'] = hausdorff_distance
            custom_objects['combined_loss'] = combined_loss()

        best_model = tf.keras.models.load_model(fold_model_path, custom_objects=custom_objects)

        test_results = best_model.evaluate(X_test, y_test, verbose=0)
        fold_results.append(test_results)

        print(f"Fold {fold_num + 1} Test Results:")
        metric_names = ['Loss', 'Mean IoU', 'Dice Coef', 'Recall', 'Precision']
        if CONFIG['MODE'] == 'enhanced':
            metric_names.append('Hausdorff Dist')

        for i, (name, value) in enumerate(zip(metric_names, test_results)):
            print(f"  {name}: {value:.4f}")

        # Save sample predictions
        if CONFIG['SAVE_RESULTS']:
            sample_idx = np.random.randint(0, len(X_test))
            sample_image = X_test[sample_idx]
            sample_mask = y_test[sample_idx]

            # Get prediction
            sample_pred = best_model.predict(np.expand_dims(sample_image, axis=0))[0]

            # Apply post-processing if enhanced mode
            if CONFIG['MODE'] == 'enhanced':
                processed_pred = post_process_prediction(sample_pred, sample_image)
            else:
                processed_pred = (sample_pred > 0.5).astype(np.float32)

            # Generate uncertainty if enabled
            uncertainty_map = None
            if CONFIG['ENABLE_UNCERTAINTY']:
                mean_pred, uncertainty_map = monte_carlo_dropout_prediction(best_model, sample_image, n_samples=15)

            # Visualize and save
            if CONFIG['MODE'] == 'enhanced':
                save_path = os.path.join(results_dir, f'fold_{fold_num + 1}_sample_prediction.png')
                visualize_segmentation_comparison(
                    sample_image, sample_mask, processed_pred, uncertainty_map,
                    title=f"Fold {fold_num + 1} - Sample Prediction",
                    save_path=save_path
                )

                # Generate and save Grad-CAM
                try:
                    grad_cam_heatmap = generate_grad_cam(best_model, sample_image, layer_name='conv2d_4')
                    grad_cam_path = os.path.join(results_dir, f'fold_{fold_num + 1}_grad_cam.png')
                    visualize_grad_cam(sample_image, grad_cam_heatmap, save_path=grad_cam_path)
                except Exception as e:
                    print(f"Grad-CAM generation failed: {e}")
            else:
                # Standard visualization for basic mode
                visualize_sample(sample_image, sample_mask, processed_pred,
                               title=f"Fold {fold_num + 1} - Sample Prediction")

    # Summarize results
    print("\n--- Training Complete: Summary of Results ---")
    fold_results_np = np.array(fold_results)

    print("\nAverage Test Performance Across All Folds:")
    print("Metric\t\tAverage\t\tStd Dev")

    for i, metric in enumerate(metric_names):
        avg = np.mean(fold_results_np[:, i])
        std = np.std(fold_results_np[:, i])
        print(f"{metric}\t\t{avg:.4f}\t\t±{std:.4f}")

    # Save training history plot for last fold
    if histories and CONFIG['SAVE_RESULTS']:
        history_path = os.path.join(results_dir, 'training_history.png')
        plot_training_history(histories[-1], title=f"Training History (Last Fold) - {CONFIG['MODE'].upper()} MODE", save_path=history_path)

    # Save metrics comparison for enhanced mode
    if CONFIG['MODE'] == 'enhanced' and CONFIG['SAVE_RESULTS']:
        metrics_dict = {}
        for i, result in enumerate(fold_results):
            metrics_dict[f'Fold {i+1}'] = {
                'Loss': result[0],
                'Dice Coef': result[2],
                'Recall': result[3],
                'Precision': result[4]
            }

        metrics_path = os.path.join(results_dir, 'metrics_comparison.png')
        try:
            from visualization.visualization_utils import visualize_metrics_comparison
            visualize_metrics_comparison(metrics_dict, title="Metrics Across Folds", save_path=metrics_path)
        except:
            print("Metrics comparison visualization not available in standard mode")

    print(f"\nAll results saved to: {results_dir}")
    print(f"{CONFIG['MODE'].upper()} training completed successfully!")

if __name__ == '__main__':
    main()
