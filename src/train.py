import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import KFold, train_test_split # Added train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold # New import!

# Import your data_loader and unet_model functions
from dataloader import load_isic_data, visualize_sample # Removed split_data as we'll handle splitting here
from unet_model import unet_model


# --- Define Custom Metrics/Losses ---
# Dice Coefficient (F1 Score) - commonly used for segmentation
def dice_coef(y_true, y_pred, smooth=1e-7): #
    # Flatten y_true and y_pred to 1D arrays
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32) #
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32) #

    # Calculate intersection
    intersection = tf.reduce_sum(y_true_f * y_pred_f) #

    # Calculate sum of squares (or just sum for binary)
    sum_true = tf.reduce_sum(y_true_f) #
    sum_pred = tf.reduce_sum(y_pred_f) #

    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (sum_true + sum_pred + smooth) #
    return dice


# Dice Loss - directly optimizes 1 - Dice Coefficient
def dice_loss(y_true, y_pred): #
    return 1 - dice_coef(y_true, y_pred) #


# Combined BCE and Dice Loss - often performs well for segmentation
def bce_dice_loss(y_true, y_pred): #
    bce = BinaryCrossentropy()(y_true, y_pred)  # Standard Binary Cross-Entropy
    dice = dice_loss(y_true, y_pred)  # Custom Dice Loss
    return bce + dice  # Simple sum, you can weight them if needed (e.g., 0.5*bce + 0.5*dice)


# --- Main Training Script ---
if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_HEIGHT = 128 #
    IMAGE_WIDTH = 128 #
    NUM_CHANNELS = 3  # RGB images
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS) #

    # Path to your data (ensure this matches your actual data location)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #
    DATA_DIR = os.path.join(project_root, 'data', 'isic_2016') #

    # Training hyperparameters
    BATCH_SIZE = 16 #
    EPOCHS = 50 #
    LEARNING_RATE = 1e-4 #

    NUM_FOLDS = 5 # New parameter for K-Fold

    # --- 1. Load Data ---
    print("\n--- Loading Data ---") #
    images, masks, image_ids = load_isic_data(DATA_DIR, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH,
                                              apply_augmentation=True) #

    if images is None or masks is None: #
        print("Data loading failed. Exiting training script.") #
        exit() #

    # First, split off the final test set (20% of original data)
    # The remaining 80% will be used for K-Fold cross-validation
    X_train_val, X_test, y_train_val, y_test = train_test_split( #
        images, masks, test_size=0.2, random_state=42 #
    )
    print(f"Initial split: Training+Validation ({len(X_train_val)} samples), Test ({len(X_test)} samples)") #

    # Prepare for K-Fold Cross-Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42) #

    fold_results = [] # To store test metrics for each fold
    histories = [] # To store training histories for each fold
    best_models_paths = [] # To store paths of best models from each fold

    print(f"\n--- Starting K-Fold Cross-Validation with {NUM_FOLDS} Folds ---") #

    for fold_num, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)): #
        print(f"\n----- Fold {fold_num + 1}/{NUM_FOLDS} -----") #

        X_train_fold, X_val_fold = X_train_val[train_index], X_train_val[val_index] #
        y_train_fold, y_val_fold = y_train_val[train_index], y_train_val[val_index] #

        print(f"Fold {fold_num + 1} data: Train ({len(X_train_fold)} samples), Validation ({len(X_val_fold)} samples)") #

        # --- 2. Build Model (fresh model for each fold) ---
        tf.keras.backend.clear_session() # Clear previous Keras session to ensure fresh model each fold
        model = unet_model(input_size=INPUT_SHAPE) #

        # --- 3. Compile Model ---
        optimizer = Adam(learning_rate=LEARNING_RATE) #
        loss_function = bce_dice_loss #
        metrics = [ #
            MeanIoU(num_classes=2, name='mean_io_u'), #
            dice_coef, #
            tf.keras.metrics.Recall(name='recall'), #
            tf.keras.metrics.Precision(name='precision') #
        ]
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics) #
        print("Model compiled for this fold.") #

        # --- 4. Callbacks (specific for this fold) ---
        fold_model_save_dir = os.path.join(project_root, 'models', f'fold_{fold_num + 1}') #
        os.makedirs(fold_model_save_dir, exist_ok=True) #
        fold_model_checkpoint_path = os.path.join(fold_model_save_dir, 'unet_isic_best_model_fold_{}.keras'.format(fold_num + 1)) #
        best_models_paths.append(fold_model_checkpoint_path) # Store path for later

        fold_callbacks = [ #
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), #
            ModelCheckpoint(fold_model_checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0), # verbose=0 to reduce clutter
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0) # verbose=0 to reduce clutter
        ]
        print("Callbacks set up for this fold.") #

        # --- 5. Train Model ---
        print(f"Starting training for Fold {fold_num + 1}...") #
        history = model.fit( #
            X_train_fold, y_train_fold, #
            batch_size=BATCH_SIZE, #
            epochs=EPOCHS, #
            validation_data=(X_val_fold, y_val_fold), #
            callbacks=fold_callbacks, #
            verbose=1 # Show progress bar and metrics per epoch
        )
        print(f"Training finished for Fold {fold_num + 1}!") #
        histories.append(history) #

        # Evaluate the best model of this fold on the fixed TEST set
        print(f"Evaluating Fold {fold_num + 1} best model on the Test Set...") #
        best_model_fold = tf.keras.models.load_model( #
            fold_model_checkpoint_path, #
            custom_objects={ #
                'dice_coef': dice_coef, #
                'dice_loss': dice_loss, #
                'bce_dice_loss': bce_dice_loss #
            }
        )
        test_metrics = best_model_fold.evaluate(X_test, y_test, verbose=0) #
        fold_results.append(test_metrics) #
        print(f"Fold {fold_num + 1} Test Metrics: Loss={test_metrics[0]:.4f}, Dice={test_metrics[2]:.4f}") #


    print("\n--- K-Fold Cross-Validation Finished ---") #

    # --- Summarize K-Fold Results ---
    print("\n--- K-Fold Test Set Performance Summary ---") #
    # Convert fold_results to numpy array for easier calculations
    fold_results_np = np.array(fold_results) #

    # Extract metrics by index (assuming order from model.compile)
    avg_loss = np.mean(fold_results_np[:, 0]) #
    std_loss = np.std(fold_results_np[:, 0]) #
    avg_mean_io_u = np.mean(fold_results_np[:, 1]) #
    std_mean_io_u = np.std(fold_results_np[:, 1]) #
    avg_dice_coef = np.mean(fold_results_np[:, 2]) #
    std_dice_coef = np.std(fold_results_np[:, 2]) #
    avg_recall = np.mean(fold_results_np[:, 3]) #
    std_recall = np.std(fold_results_np[:, 3]) #
    avg_precision = np.mean(fold_results_np[:, 4]) #
    std_precision = np.std(fold_results_np[:, 4]) #

    print(f"Average Test Loss: {avg_loss:.4f} (+/- {std_loss:.4f})") #
    print(f"Average Test Mean IoU: {avg_mean_io_u:.4f} (+/- {std_mean_io_u:.4f})") #
    print(f"Average Test Dice Coefficient: {avg_dice_coef:.4f} (+/- {std_dice_coef:.4f})") #
    print(f"Average Test Recall: {avg_recall:.4f} (+/- {std_recall:.4f})") #
    print(f"Average Test Precision: {avg_precision:.4f} (+/- {std_precision:.4f})") #

    # --- Plotting Training History for each fold (optional, can be very many plots) ---
    # For a CV, you might just show one representative plot or average curves.
    # We will just plot the history of the *last* fold for simplicity in this script.
    print("\n--- Plotting Training History (Last Fold) ---") #
    if histories: #
        last_history = histories[-1] # Plot history of the last trained fold
        plt.figure(figsize=(14, 6)) #

        plt.subplot(1, 2, 1) #
        plt.plot(last_history.history['loss'], label='Train Loss', color='blue') #
        plt.plot(last_history.history['val_loss'], label='Validation Loss', color='red') #
        plt.title('Loss over Epochs (Last Fold)', fontsize=14) #
        plt.xlabel('Epoch', fontsize=12) #
        plt.ylabel('Loss', fontsize=12) #
        plt.legend(fontsize=10) #
        plt.grid(True, linestyle='--', alpha=0.7) #

        plt.subplot(1, 2, 2) #
        plt.plot(last_history.history['dice_coef'], label='Train Dice Coef', color='green') #
        plt.plot(last_history.history['val_dice_coef'], label='Validation Dice Coef', color='orange') #
        plt.title('Dice Coefficient over Epochs (Last Fold)', fontsize=14) #
        plt.xlabel('Epoch', fontsize=12) #
        plt.ylabel('Dice Coefficient', fontsize=12) #
        plt.legend(fontsize=10) #
        plt.grid(True, linestyle='--', alpha=0.7) #

        plt.tight_layout() #
        plt.show() #
    else: #
        print("No training histories to plot.") #

    # --- 7. Visualize Predictions (Optional) ---
    print("\n--- Visualizing Predictions on Test Samples ---") #
    num_predictions_to_show = 5 #
    # Load the best model from the LAST fold for visualization
    # Or you could load the model that had the best average performance across folds, if you implemented that.
    if best_models_paths: #
        final_best_model_path = best_models_paths[-1] # Path to the best model from the last fold
        best_model_for_viz = tf.keras.models.load_model( #
            final_best_model_path, #
            custom_objects={ #
                'dice_coef': dice_coef, #
                'dice_loss': dice_loss, #
                'bce_dice_loss': bce_dice_loss #
            }
        )
        for i in range(num_predictions_to_show): #
            idx = np.random.randint(0, len(X_test)) #
            sample_image = X_test[idx] #
            sample_mask = y_test[idx] #

            sample_image_batch = np.expand_dims(sample_image, axis=0) #
            prediction = best_model_for_viz.predict(sample_image_batch, verbose=0)[0] #
            predicted_mask = (prediction > 0.5).astype(np.float32) #

            visualize_sample(sample_image, sample_mask, predicted_mask, title=f"Test Sample {idx} Prediction (from Last Fold's Best Model)") #
    else: #
        print("No models trained or saved for visualization.") #

    print("\n--- Training and Evaluation Complete ---") #