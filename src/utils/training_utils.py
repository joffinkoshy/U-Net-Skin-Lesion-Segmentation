import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.optimizers import Adam, AdamW, NADAM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa

def cosine_annealing_schedule(epoch, lr, max_epochs=100, warmup_epochs=10):
    """
    Cosine annealing learning rate schedule with warmup.

    Args:
        epoch: Current epoch
        lr: Initial learning rate
        max_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        Adjusted learning rate
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        return lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))

def get_optimizer(optimizer_name='adamw', learning_rate=1e-4):
    """
    Get optimizer based on name.

    Args:
        optimizer_name: Name of optimizer ('adam', 'adamw', 'nadam')
        learning_rate: Initial learning rate

    Returns:
        Configured optimizer
    """
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        return AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    elif optimizer_name == 'nadam':
        return NADAM(learning_rate=learning_rate)
    else:
        return Adam(learning_rate=learning_rate)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.

    Args:
        gamma: Focusing parameter
        alpha: Balancing parameter

    Returns:
        Focal loss function
    """
    def loss(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Calculate focal loss
        y_true = tf.cast(y_true, tf.float32)
        at = alpha * y_true + (1 - alpha) * (1 - y_true)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = at * (1 - pt) ** gamma * bce

        return tf.reduce_mean(fl)
    return loss

def combined_loss(bce_weight=0.5, dice_weight=0.5, focal_weight=0.0):
    """
    Combined loss function with BCE, Dice, and optional Focal loss.

    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss

    Returns:
        Combined loss function
    """
    def loss(y_true, y_pred):
        # Binary Cross Entropy
        bce = BinaryCrossentropy()(y_true, y_pred)

        # Dice Loss
        dice = dice_loss(y_true, y_pred)

        # Focal Loss (optional)
        if focal_weight > 0:
            fl = focal_loss()(y_true, y_pred)
            return bce_weight * bce + dice_weight * dice + focal_weight * fl
        else:
            return bce_weight * bce + dice_weight * dice
    return loss

def dice_coef(y_true, y_pred, smooth=1e-7):
    """Dice coefficient for segmentation evaluation."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    sum_true = tf.reduce_sum(y_true_f)
    sum_pred = tf.reduce_sum(y_pred_f)

    dice = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return dice

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation."""
    return 1 - dice_coef(y_true, y_pred)

def hausdorff_distance(y_true, y_pred, threshold=0.5):
    """
    Hausdorff distance for boundary assessment.

    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        threshold: Threshold for binary conversion

    Returns:
        Hausdorff distance
    """
    # Convert to binary masks
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)

    # Find contours (simplified version)
    true_contour = tf.where(y_true_binary == 1)
    pred_contour = tf.where(y_pred_binary == 1)

    # Calculate distances
    if tf.size(true_contour) > 0 and tf.size(pred_contour) > 0:
        # Calculate pairwise distances
        true_expanded = tf.expand_dims(true_contour, 1)
        pred_expanded = tf.expand_dims(pred_contour, 0)

        distances = tf.norm(true_expanded - pred_expanded, axis=2)

        # Hausdorff distance
        h_true = tf.reduce_max(tf.reduce_min(distances, axis=1))
        h_pred = tf.reduce_max(tf.reduce_min(distances, axis=0))
        return tf.reduce_max([h_true, h_pred])
    else:
        return tf.constant(0.0, dtype=tf.float32)

def get_callbacks(model_save_path, patience=15, reduce_lr_patience=7):
    """
    Get training callbacks.

    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction

    Returns:
        List of callbacks
    """
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=reduce_lr_patience,
                         min_lr=1e-7, verbose=1),
        LearningRateScheduler(cosine_annealing_schedule, verbose=1)
    ]

def enable_mixed_precision():
    """Enable mixed precision training for better performance."""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision training enabled")

def monte_carlo_dropout_prediction(model, image, n_samples=10):
    """
    Monte Carlo dropout for uncertainty estimation.

    Args:
        model: Trained model with dropout layers
        image: Input image
        n_samples: Number of samples for MC dropout

    Returns:
        Mean prediction and uncertainty map
    """
    predictions = []
    for _ in range(n_samples):
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions.append(pred[0])

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)

    return mean_prediction, uncertainty

if __name__ == '__main__':
    # Test the utilities
    print("Testing training utilities...")

    # Test loss functions
    y_true = np.random.randint(0, 2, (2, 128, 128, 1)).astype(np.float32)
    y_pred = np.random.uniform(0, 1, (2, 128, 128, 1)).astype(np.float32)

    combined = combined_loss()(y_true, y_pred)
    print(f"Combined loss: {combined.numpy()}")

    dice = dice_coef(y_true, y_pred)
    print(f"Dice coefficient: {dice.numpy()}")

    print("Training utilities test completed!")
