import numpy as np
import cv2
from skimage.morphology import binary_opening, binary_closing, remove_small_objects
from skimage.measure import label, regionprops

def morphological_post_processing(prediction, threshold=0.5, min_area=50):
    """
    Apply morphological operations to clean up segmentation masks.

    Args:
        prediction: Model prediction (probability map)
        threshold: Threshold for binary conversion
        min_area: Minimum area for connected components

    Returns:
        Processed binary mask
    """
    # Convert to binary mask
    binary = (prediction > threshold).astype(np.uint8)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)

    # Remove small artifacts (opening)
    cleaned = binary_opening(binary, kernel)

    # Fill small holes (closing)
    smoothed = binary_closing(cleaned, kernel)

    # Remove small connected components
    labeled = label(smoothed)
    if np.max(labeled) > 0:
        # Remove small regions
        processed = remove_small_objects(labeled, min_size=min_area)
        # Keep only the largest component if multiple exist
        regions = regionprops(labeled)
        if len(regions) > 1:
            largest_region = max(regions, key=lambda r: r.area)
            processed = (labeled == largest_region.label).astype(np.uint8)
        else:
            processed = processed.astype(np.uint8)
    else:
        processed = smoothed

    return processed.astype(np.float32)

def contour_smoothing(mask, sigma=1.5):
    """
    Smooth mask contours using Gaussian filtering.

    Args:
        mask: Binary mask
        sigma: Standard deviation for Gaussian filter

    Returns:
        Smoothed mask
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Create empty mask
        smoothed = np.zeros_like(mask)

        # Draw smoothed contours
        for contour in contours:
            # Approximate contour with fewer points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw filled contour
            cv2.drawContours(smoothed, [approx], -1, 1, thickness=cv2.FILLED)

        return smoothed
    else:
        return mask

def multi_scale_post_processing(prediction, scales=[0.5, 1.0, 2.0]):
    """
    Multi-scale post-processing for improved segmentation.

    Args:
        prediction: Original prediction
        scales: List of scales to process

    Returns:
        Enhanced prediction
    """
    enhanced = np.zeros_like(prediction)

    for scale in scales:
        # Resize prediction
        if scale != 1.0:
            resized = cv2.resize(prediction, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # Process at this scale
            processed = morphological_post_processing(resized)
            # Resize back
            processed = cv2.resize(processed, prediction.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        else:
            processed = morphological_post_processing(prediction)

        # Accumulate results
        enhanced += processed

    # Average results
    enhanced = enhanced / len(scales)

    return enhanced

def boundary_refinement(mask, original_image, sigma=1.0):
    """
    Refine mask boundaries using image gradients.

    Args:
        mask: Binary mask
        original_image: Original RGB image
        sigma: Standard deviation for Gaussian blur

    Returns:
        Refined mask
    """
    # Convert to grayscale if needed
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_image.copy()

    # Compute image gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

    # Find mask boundaries
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Create distance transform
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)

        # Create refined mask
        refined = np.zeros_like(mask)

        # Use gradient information near boundaries
        boundary_width = 5
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if dist_transform[i, j] < boundary_width:
                    # Near boundary - use gradient information
                    if gradient_magnitude[i, j] > 0.3:  # High gradient indicates edge
                        refined[i, j] = 1.0
                    else:
                        refined[i, j] = mask[i, j]
                else:
                    # Far from boundary - keep original
                    refined[i, j] = mask[i, j]

        return refined
    else:
        return mask

def post_process_prediction(prediction, original_image=None, advanced=True):
    """
    Complete post-processing pipeline for model predictions.

    Args:
        prediction: Model prediction (probability map)
        original_image: Original image for boundary refinement (optional)
        advanced: Whether to use advanced processing

    Returns:
        Processed binary mask
    """
    if advanced and original_image is not None:
        # Full pipeline with boundary refinement
        processed = morphological_post_processing(prediction)
        processed = contour_smoothing(processed)
        processed = boundary_refinement(processed, original_image)
    else:
        # Basic pipeline
        processed = morphological_post_processing(prediction)
        processed = contour_smoothing(processed)

    return processed

if __name__ == '__main__':
    # Test post-processing functions
    print("Testing post-processing utilities...")

    # Create test data
    test_pred = np.random.uniform(0, 1, (128, 128))
    test_pred[40:80, 40:80] = 0.8  # Create a square region

    test_image = np.random.uniform(0, 1, (128, 128, 3))

    # Test basic processing
    basic_result = morphological_post_processing(test_pred)
    print(f"Basic processing result shape: {basic_result.shape}")

    # Test advanced processing
    advanced_result = post_process_prediction(test_pred, test_image)
    print(f"Advanced processing result shape: {advanced_result.shape}")

    print("Post-processing utilities test completed!")
