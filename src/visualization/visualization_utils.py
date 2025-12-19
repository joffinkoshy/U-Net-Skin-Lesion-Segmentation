import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_custom_colormap():
    """Create a custom colormap for better visualization."""
    colors = ['black', 'red', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('custom_heat', colors)

def visualize_segmentation_comparison(image, ground_truth, prediction, uncertainty=None,
                                    title="Segmentation Comparison", save_path=None):
    """
    Visualize image, ground truth, prediction, and uncertainty.

    Args:
        image: Original image
        ground_truth: Ground truth mask
        prediction: Predicted mask
        uncertainty: Uncertainty map (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Create custom colormap
    cmap = create_custom_colormap()

    if uncertainty is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth overlay
    axes[1].imshow(image)
    gt_overlay = axes[1].imshow(ground_truth.squeeze(), alpha=0.5, cmap='viridis')
    axes[1].set_title('Ground Truth Overlay')
    axes[1].axis('off')

    # Add colorbar for ground truth
    if uncertainty is not None:
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(gt_overlay, cax=cax, orientation='vertical')

    # Prediction overlay
    axes[2].imshow(image)
    pred_overlay = axes[2].imshow(prediction.squeeze(), alpha=0.5, cmap='magma')
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')

    # Add colorbar for prediction
    if uncertainty is not None:
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(pred_overlay, cax=cax, orientation='vertical')

    if uncertainty is not None:
        # Uncertainty map
        axes[3].imshow(uncertainty.squeeze(), cmap=cmap)
        axes[3].set_title('Uncertainty Map')
        axes[3].axis('off')

        # Add colorbar for uncertainty
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(axes[3].images[0], cax=cax, orientation='vertical')

        # Uncertainty overlay
        axes[4].imshow(image)
        unc_overlay = axes[4].imshow(uncertainty.squeeze(), alpha=0.5, cmap=cmap)
        axes[4].set_title('Uncertainty Overlay')
        axes[4].axis('off')

        # Add colorbar for uncertainty overlay
        divider = make_axes_locatable(axes[4])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(unc_overlay, cax=cax, orientation='vertical')

        # Remove unused subplot
        fig.delaxes(axes[5])

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()

def generate_grad_cam(model, image, layer_name='conv2d_4', class_index=0):
    """
    Generate Grad-CAM visualization to explain model decisions.

    Args:
        model: Trained model
        image: Input image
        layer_name: Name of layer to generate Grad-CAM for
        class_index: Class index for visualization

    Returns:
        Grad-CAM heatmap
    """
    # Create model that outputs target layer activations and predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        if len(predictions.shape) > 3:
            loss = predictions[:, :, :, class_index]
        else:
            loss = predictions[:, class_index]

    # Get gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activations by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to only keep positive influences
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap
    heatmap /= tf.math.reduce_max(heatmap) + 1e-7

    return heatmap.numpy()

def visualize_grad_cam(image, heatmap, alpha=0.5, title="Grad-CAM Visualization", save_path=None):
    """
    Visualize Grad-CAM heatmap overlay on original image.

    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency for overlay
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Create heatmap visualization
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert heatmap to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on image
    superimposed_img = heatmap * alpha + image * 255 * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = superimposed_img / 255.0

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap / 255.0)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(superimposed_img)
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to: {save_path}")

    plt.show()

def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training and validation metrics from training history.

    Args:
        history: Training history object
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice Coefficient
    if 'dice_coef' in history.history:
        axes[0, 1].plot(history.history['dice_coef'], label='Train Dice')
        axes[0, 1].plot(history.history['val_dice_coef'], label='Validation Dice')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Mean IoU
    if 'mean_io_u' in history.history:
        axes[0, 2].plot(history.history['mean_io_u'], label='Train Mean IoU')
        axes[0, 2].plot(history.history['val_mean_io_u'], label='Validation Mean IoU')
        axes[0, 2].set_title('Mean IoU')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Mean IoU')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Precision
    if 'precision' in history.history:
        axes[1, 2].plot(history.history['precision'], label='Train Precision')
        axes[1, 2].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 2].set_title('Precision')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Precision')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    # Remove unused subplots
    for i in range(2):
        for j in range(3):
            if not any(axes[i, j].get_lines()):
                fig.delaxes(axes[i, j])

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")

    plt.show()

def visualize_metrics_comparison(metrics_dict, title="Metrics Comparison", save_path=None):
    """
    Visualize comparison of different metrics across models/folds.

    Args:
        metrics_dict: Dictionary of metrics {model_name: {metric: value}}
        title: Plot title
        save_path: Path to save figure (optional)
    """
    metrics = list(next(iter(metrics_dict.values())).keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        values = [metrics_dict[name][metric] for name in metrics_dict]
        names = list(metrics_dict.keys())

        axes[i].bar(names, values)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        axes[i].set_xticklabels(names, rotation=45, ha='right')
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")

    plt.show()

if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create test data
    test_image = np.random.uniform(0, 1, (128, 128, 3))
    test_gt = np.zeros((128, 128, 1))
    test_gt[30:90, 30:90] = 1.0
    test_pred = np.zeros((128, 128, 1))
    test_pred[35:85, 35:85] = 1.0
    test_unc = np.random.uniform(0, 0.3, (128, 128, 1))

    # Test visualization
    visualize_segmentation_comparison(test_image, test_gt, test_pred, test_unc,
                                    title="Test Visualization")

    print("Visualization utilities test completed!")
