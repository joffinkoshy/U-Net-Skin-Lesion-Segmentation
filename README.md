# **U-Net for Skin Lesion Segmentation (ISIC 2016 Dataset)**

## **1\. Project Overview**

This project implements a U-Net convolutional neural network for the automated segmentation of skin lesions from dermoscopic images. Accurate segmentation of moles and other skin lesions is a crucial step in dermatological image analysis, assisting clinicians in the early diagnosis and monitoring of conditions like melanoma. The model is trained and rigorously evaluated on the ISIC 2016 dataset \[cite: train.py, dataloader.py\], demonstrating robust performance through advanced techniques like data augmentation and K-Fold Cross-Validation.

## **2\. Dataset**

The model is trained and evaluated using the **ISIC 2016: Skin Lesion Analysis Toward Melanoma Detection** dataset \[cite: train.py, dataloader.py\].

* It consists of dermoscopic images of skin lesions along with corresponding expert-annotated segmentation masks.  
* The dataset includes 900 image-mask pairs \[cite: train.py\].

**How to obtain the dataset:**

1. Download the ISIC 2016 dataset from the official ISIC Archive:  
   * **Images**: ISIC-2016\_Training\_Data.zip (contains original lesion images)  
   * **Ground Truth Masks**: ISIC-2016\_Training\_Part1\_Segmentation\_GroundTruth.zip (contains corresponding binary masks)  
2. Unzip both archives.  
3. Create a directory structure: data/isic\_2016/ in the root of your project.  
4. Place the extracted image files (e.g., ISIC\_0000000.jpg) into data/isic\_2016/images/.  
5. Place the extracted mask files (e.g., ISIC\_0000000\_segmentation.png) into data/isic\_2016/masks/.

## **3\. Methodology**

### **3.1. U-Net Architecture**

The core of this project is a U-Net, a state-of-the-art convolutional neural network widely recognized for its effectiveness in biomedical image segmentation tasks \[cite: unet\_model.py\].

* **Encoder-Decoder Structure**: The architecture consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) to enable precise localization \[cite: unet\_model.py\].  
* **Skip Connections**: Crucial skip connections between the encoder and decoder paths transfer high-resolution feature maps, helping the decoder to recover fine-grained details lost during downsampling \[cite: unet\_model.py\].  
* The model takes input images of size 128x128x3 (RGB) and outputs a single-channel segmentation mask (128x128x1) with sigmoid activation for binary classification of each pixel \[cite: unet\_model.py, train.py\].

### **3.2. Data Preprocessing & Augmentation**

Data loading and preprocessing are handled by dataloader.py \[cite: dataloader.py\].

* **Resizing & Normalization**: All images and masks are resized to 128x128 pixels. Image pixel values are normalized to the \[0, 1\] range \[cite: dataloader.py\].  
* **Data Augmentation**: To improve the model's generalization capabilities and combat overfitting (common in limited medical datasets), extensive data augmentation is applied to the training data. This includes:  
  * Random horizontal and vertical flips \[cite: dataloader.py\].  
  * Random brightness adjustments \[cite: dataloader.py\].  
  * Random contrast adjustments \[cite: dataloader.py\].  
  * Crucially, geometric transformations (flips) are applied identically to both images and their corresponding masks to maintain pixel-wise alignment \[cite: dataloader.py\].

### **3.3. Loss Function & Metrics**

* **Loss Function**: A custom combined loss function, bce\_dice\_loss, is used, which sums the Binary Cross-Entropy (BCE) loss and the Dice Loss \[cite: train.py\]. This combination is highly effective for segmentation tasks, as BCE focuses on pixel-wise accuracy, while Dice Loss emphasizes overlap and is particularly useful for imbalanced classes (small lesions).  
* **Evaluation Metrics**: The model's performance is monitored using standard segmentation metrics \[cite: train.py\]:  
  * **Mean IoU (Intersection over Union)**: A common metric for evaluating the overlap between predicted and ground truth masks.  
  * **Dice Coefficient (F1 Score)**: Another widely used metric, representing the harmonic mean of precision and recall, especially valuable for imbalanced segmentation problems.  
  * **Recall (Sensitivity)**: Measures the proportion of actual positive pixels that are correctly identified.  
  * **Precision (Positive Predictive Value)**: Measures the proportion of predicted positive pixels that are actually correct.

### **3.4. Training Strategy**

The model is trained using the Adam optimizer with a learning rate of 1×10−4 and a batch size of 16 \[cite: train.py\]. Several callbacks are employed to optimize and stabilize the training process \[cite: train.py\]:

* **Early Stopping**: Stops training if the validation loss does not improve for 10 consecutive epochs, preventing overfitting \[cite: train.py\].  
* **Model Checkpoint**: Saves the model weights whenever the validation loss achieves a new minimum, ensuring that the best-performing model is preserved \[cite: train.py\].  
* **Reduce Learning Rate on Plateau**: Reduces the learning rate by a factor of 0.5 if the validation loss does not improve for 5 consecutive epochs, helping the model escape local minima \[cite: train.py\].

### **3.5. K-Fold Cross-Validation (Robust Evaluation)**

To provide a more robust and statistically sound evaluation of the model's generalization performance, K-Fold Cross-Validation is implemented with NUM\_FOLDS \= 5 \[cite: train.py\].

* The entire dataset is first split into a **fixed, truly unseen 20% test set** and an 80% training/validation pool \[cite: train.py\].  
* The 5-Fold Cross-Validation is then performed on this 80% pool. In each fold, the data is divided into training and validation sets \[cite: train.py\].  
* A fresh U-Net model is trained for each fold, and its best weights (based on its internal validation set) are then evaluated on the **fixed 20% test set** \[cite: train.py\].  
* The final performance metrics are the average and standard deviation of the test results across all 5 folds. This minimizes bias associated with a single arbitrary data split and provides a more reliable estimate of the model's expected performance on new, unseen data.

## **4\. Results**

The model demonstrates strong performance in segmenting skin lesions, as evidenced by the quantitative metrics and qualitative visualizations.

### **4.1. K-Fold Cross-Validation Results (Average on Test Set, 5 Folds)**

The average test metrics across 5 folds, along with their standard deviations, are:

| Metric | Average Value | Standard Deviation |
| :---- | :---- | :---- |
| **Test Loss** | 0.3168 | ±0.0134 |
| **Test Mean IoU** | 0.3861 | ±0.0080 |
| **Test Dice Coef** | 0.8685 | ±0.0041 |
| **Test Recall** | 0.8662 | ±0.0047 |
| **Test Precision** | 0.9034 | ±0.0102 |

*Note: Metrics calculated on a held-out 20% test set, evaluated by the best model from each fold \[cite: train.py\].*

### **4.2. Training History (Last Fold)**

The following plots illustrate the training and validation performance over epochs for the last fold \[cite: KFold.png\]:  
\[Insert KFold.png here\]

### **4.3. Qualitative Visualizations**

The model produces highly accurate and smooth segmentation masks, closely matching the ground truth.

| Original Image | Ground Truth Mask | Predicted Mask |
| :---- | :---- | :---- |
| \[Insert Test10.jpg here\] | \[Insert Test10.jpg here\] | \[Insert Test10.jpg here\] |
| *Example showing irregular shape handling.* |  |  |
| \[Insert Test8.png here\] | \[Insert Test8.png here\] | \[Insert Test8.png here\] |
| *Example showing multiple lesion components.* |  |  |
| \[Insert Test6.jpg here\] | \[Insert Test6.jpg here\] | \[Insert Test6.jpg here\] |
| *Example showing accurate overall segmentation.* |  |  |
| \[Insert Test9.png here\] | \[Insert Test9.png here\] | \[Insert Test9.png here\] |
| *Example showing minor smoothing on irregular boundaries.* |  |  |
| \[Insert Test7.jpg here\] | \[Insert Test7.jpg here\] | \[Insert Test7.jpg here\] |
| *Example showing a more complex background. Note minor over-segmentation of background elements in prediction.* |  |  |

## **5\. Future Work**

* **Hyperparameter Optimization**: Conduct a systematic search for optimal hyperparameters (e.g., learning rate, batch size, network depth/width) using tools like Keras Tuner or Optuna.  
* **Advanced Architectures**: Experiment with U-Net variants (e.g., U-Net++, Attention U-Net) or incorporate pre-trained backbones (e.g., ResNet, EfficientNet) for feature extraction.  
* **Uncertainty Quantification**: Implement methods to quantify the model's confidence in its predictions, which is crucial in medical applications.  
* **Larger/Diverse Datasets**: Test and validate the model on larger and more diverse skin lesion datasets (e.g., ISIC 2017, 2018\) to assess scalability and generalization.

## **6\. How to Run**

1. **Clone the repository:**  
   git clone https://github.com/joffinkoshy/U-Net-Lesion-Segmentation.git  
   cd MedicalSegmentation

2. Set up the environment:  
   It's recommended to use a virtual environment.  
   python3 \-m venv venv  
   source venv/bin/activate  \# On Windows: \`.\\venv\\Scripts\\activate\`  
   pip install \-r requirements.txt

   (To create requirements.txt: Run pip freeze \> requirements.txt from your activated virtual environment)  
   (Note: Ensure you have TensorFlow installed with Metal/GPU support for Apple Silicon if applicable for faster training)  
3. Download and organize the dataset:  
   Follow the instructions in Section 2 (Dataset) above to place the ISIC 2016 images and masks in data/isic\_2016/images/ and data/isic\_2016/masks/ respectively.  
4. **Run the training script:**  
   python src/train.py

   This script will perform the K-Fold cross-validation, print the aggregated results, display training history plots, and show sample predictions.

Author: Joffin Koshy  
Date: June 2025
