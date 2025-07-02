# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:52:30 2025

@author: VIJAY
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix # Not directly used for Dice/IoU but useful for debugging segmentation

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# --- Constants for Image Processing and Model Training ---
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1 # Original images are grayscale
NUM_CLASSES = 3  # 0: Background, 1: Cell Border, 2: Cell Interior (from Final_Project.ipynb analysis)
BATCH_SIZE = 32   # Smaller batch size often good for U-Net due to memory
EPOCHS = 50      # Initial number of epochs; EarlyStopping will manage this

# --- 1. Data Loading and Preprocessing for Semantic Segmentation ---
def load_segmentation_data(original_images_dir, mask_images_dir, img_width, img_height, num_classes):
    """
    Loads and preprocesses original images and their corresponding ground-truth masks
    for semantic segmentation.

    Args:
        original_images_dir (str): Path to the directory containing original (raw) images.
        mask_images_dir (str): Path to the directory containing ground-truth masks (labels).
        img_width (int): Desired width to resize images and masks to.
        img_height (int): Desired height to resize images and masks to.
        num_classes (int): Number of segmentation classes (e.g., 3 for bg, border, interior).

    Returns:
        tuple: A tuple (X, Y, filenames) where X is a NumPy array of preprocessed images,
               Y is a NumPy array of corresponding one-hot encoded masks, and filenames is a list.
    """
    X = [] # List to store preprocessed original images
    Y = [] # List to store preprocessed one-hot encoded masks
    filenames = [] # List to store original image filenames for matching/tracking

    print(f"--- Loading images from: {original_images_dir} and masks from: {mask_images_dir} ---")

    original_image_paths = sorted(glob(os.path.join(original_images_dir, '*.tif')))
    mask_image_paths_dict = {os.path.basename(p): p for p in glob(os.path.join(mask_images_dir, '*.tif'))}

    if not original_image_paths or not mask_image_paths_dict:
        print("Error: Could not find original or mask image files. Check paths and file types.")
        return np.array([]), np.array([]), []

    for original_path in original_image_paths:
        filename = os.path.basename(original_path)
        
        # Assume mask has same filename as original, check if it exists in mask_images_dir
        if filename not in mask_image_paths_dict:
            print(f"Warning: No corresponding mask found for {filename}. Skipping.")
            continue
        
        mask_path = mask_image_paths_dict[filename]

        # Load original image (grayscale)
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        # Load mask image (color, as per your labelImages example)
        mask_img_color = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if original_img is None:
            print(f"Warning: Could not load original image {original_path}. Skipping.")
            continue
        if mask_img_color is None:
            print(f"Warning: Could not load mask image {mask_path}. Skipping.")
            continue

        # Preprocess original image
        original_img = cv2.resize(original_img, (img_width, img_height))
        original_img = original_img / 255.0 # Normalize to [0, 1]
        original_img = np.expand_dims(original_img, axis=-1) # Add channel dimension

        # Preprocess mask image
        # Convert BGR mask to a single-channel label mask (0:bg, 1:border, 2:interior)
        # Based on Final_Project.ipynb and corneacrop-1-3-1.jpg analysis:
        # Red: background, Green: cell interior, Blue: cell border
        
        # Define color ranges (B, G, R) - these are crucial and may need tuning
        # The ranges must be specific enough to distinguish between red, green, and blue pixels.
        lower_red_bg = np.array([0, 0, 100]) # Darker red background
        upper_red_bg = np.array([100, 100, 255]) # Lighter red background
        
        lower_green_cell = np.array([0, 100, 0])
        upper_green_cell = np.array([100, 255, 100]) # Allows some blue/red variation in green
        
        lower_blue_border = np.array([100, 0, 0])
        upper_blue_border = np.array([255, 100, 100]) # Allows some green/red variation in blue

        # Create binary masks for each component
        mask_bg_bin = cv2.inRange(mask_img_color, lower_red_bg, upper_red_bg)
        mask_cell_bin = cv2.inRange(mask_img_color, lower_green_cell, upper_green_cell)
        mask_border_bin = cv2.inRange(mask_img_color, lower_blue_border, upper_blue_border)

        # Combine into a single integer label mask:
        # Initialize with background (0)
        label_mask = np.zeros(mask_bg_bin.shape, dtype=np.uint8)
        # Mark cell interior pixels as 2
        label_mask[mask_cell_bin > 0] = 2 
        # Mark cell border pixels as 1 (prioritize borders if they overlap with interior for some reason)
        label_mask[mask_border_bin > 0] = 1 
        # Background remains 0

        # Resize label mask using nearest neighbor interpolation to preserve discrete labels
        label_mask_resized = cv2.resize(label_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        # One-hot encode the mask for U-Net's categorical cross-entropy loss
        # Shape will be (height, width, num_classes)
        label_mask_one_hot = to_categorical(label_mask_resized, num_classes=num_classes)

        X.append(original_img)
        Y.append(label_mask_one_hot)
        filenames.append(filename)

    print(f"--- Loaded {len(X)} image-mask pairs for segmentation training ---")
    return np.array(X), np.array(Y), filenames

# --- 2. Build U-Net Model Architecture ---
def build_unet_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), num_classes=NUM_CLASSES):
    """
    Builds a U-Net model for semantic segmentation.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes for segmentation.

    Returns:
        tf.keras.Model: Compiled Keras U-Net model.
    """
    inputs = Input(input_shape)

    # Encoder (Contracting Path)
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder (Expansive Path)
    # Block 6 (upsample from bottleneck to conv4)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3) # Skip connection
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    # Block 7 (upsample from conv6 to conv3)
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3) # Skip connection
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    # Block 8 (upsample from conv7 to conv2)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3) # Skip connection
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    # Block 9 (upsample from conv8 to conv1)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3) # Skip connection
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output Layer
    # Final 1x1 convolution with 'softmax' for multi-class pixel classification
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    return model

# --- 3. Evaluation Metrics: Dice Coefficient and IoU ---
def dice_coefficient(y_true, y_pred):
    """
    Calculates the Dice Coefficient (F1-score) for segmentation.
    Args:
        y_true (np.array): Ground truth mask (binary or integer labels for a single class).
        y_pred (np.array): Predicted mask (binary or integer labels for a single class).
    Returns:
        float: Dice coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    sum_of_areas = np.sum(y_true) + np.sum(y_pred)
    if sum_of_areas == 0:
        return 1.0 # Perfect score if no foreground present and none predicted
    return (2. * intersection) / sum_of_areas

def iou_score(y_true, y_pred):
    """
    Calculates the Intersection over Union (IoU) score for segmentation.
    Args:
        y_true (np.array): Ground truth mask (binary or integer labels for a single class).
        y_pred (np.array): Predicted mask (binary or integer labels for a single class).
    Returns:
        float: IoU score.
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 1.0 # Perfect score if no foreground present and none predicted
    return intersection / union

def evaluate_segmentation_performance(model, X_test, Y_test, num_classes):
    """
    Evaluates the segmentation model's performance using Dice Coefficient and IoU.

    Args:
        model (tf.keras.Model): Trained segmentation model.
        X_test (np.array): Test set images.
        Y_test (np.array): True one-hot encoded test masks.
        num_classes (int): Number of segmentation classes.

    Returns:
        tuple: (list of Dice scores per class, list of IoU scores per class)
    """
    print("\n--- Evaluating Segmentation Quality on Test Set ---")
    Y_pred_proba = model.predict(X_test)
    Y_pred_labels = np.argmax(Y_pred_proba, axis=-1) # Convert probabilities to class labels (H, W)

    dice_scores = []
    iou_scores = []
    class_names = {0: "Background", 1: "Cell Border", 2: "Cell Interior"} # For clear output

    # Calculate metrics per class
    for class_id in range(num_classes):
        # Extract the binary mask for the current class from one-hot encoded Y_test
        y_true_class = (np.argmax(Y_test, axis=-1) == class_id).astype(np.float32)
        # Extract the binary mask for the current class from predicted labels
        y_pred_class = (Y_pred_labels == class_id).astype(np.float32)

        dice = dice_coefficient(y_true_class, y_pred_class)
        iou = iou_score(y_true_class, y_pred_class)
        
        dice_scores.append(dice)
        iou_scores.append(iou)

        print(f"Class {class_id} ({class_names.get(class_id, 'Unknown')}): Dice Coeff: {dice:.4f}, IoU: {iou:.4f}")
    
    print(f"Mean Dice Coefficient (overall): {np.mean(dice_scores):.4f}")
    print(f"Mean IoU Score (overall): {np.mean(iou_scores):.4f}")

    return dice_scores, iou_scores


# --- 4. Post-processing and Deep Morphological Analysis ---
def post_process_and_analyze_morphology(predicted_masks_labels, original_images_display, ground_truth_masks_labels_display, filenames, num_samples_to_analyze=3, save_plots_dir=None):
    """
    Applies post-processing to predicted masks and calculates morphological metrics.
    Also visualizes original, ground truth, and predicted masks, saving them to a directory.

    Args:
        predicted_masks_labels (np.array): Predicted masks as integer labels (H, W).
        original_images_display (np.array): Original images for display.
        ground_truth_masks_labels_display (np.array): True labels for display.
        filenames (list): List of filenames corresponding to the images.
        num_samples_to_analyze (int): Number of samples to analyze and display.
        save_plots_dir (str, optional): Directory to save the visualization plots.
                                        If None, plots will be displayed interactively.
    """
    print(f"\n--- Performing Morphological Analysis on {num_samples_to_analyze} Predicted Masks ---")

    if predicted_masks_labels.shape[0] == 0:
        print("No predicted masks available for morphological analysis.")
        return

    if save_plots_dir:
        os.makedirs(save_plots_dir, exist_ok=True)
        print(f"Saving segmentation visualizations to: {save_plots_dir}")

    # Select samples for analysis (ensure we don't go out of bounds)
    num_actual_samples = min(num_samples_to_analyze, predicted_masks_labels.shape[0])

    all_cell_areas = []
    all_hexagonal_counts = 0
    total_cells_analyzed = 0

    for i in range(num_actual_samples):
        current_predicted_mask = predicted_masks_labels[i]
        current_original_img = original_images_display[i].squeeze() # Remove channel dim for display
        current_ground_truth_mask = np.argmax(ground_truth_masks_labels_display[i], axis=-1) # Convert one-hot to labels
        current_filename = filenames[i]

        print(f"\nAnalyzing sample {i+1}/{num_actual_samples} (Image: {current_filename})...")
        
        # --- Create a binary mask of cell interior + border (foreground) ---
        # Combine class 1 (border) and class 2 (interior) into a single foreground mask (white = 255).
        # Background (class 0) remains black (0).
        combined_cells_mask = np.zeros_like(current_predicted_mask, dtype=np.uint8)
        combined_cells_mask[(current_predicted_mask == 1) | (current_predicted_mask == 2)] = 255
        
        # Optional: Apply morphological closing to fill small holes and connect small gaps
        # This helps in getting better contours for cell counting/shape analysis.
        kernel_morph = np.ones((3,3), np.uint8) # Small kernel for refinement
        combined_cells_mask_cleaned = cv2.morphologyEx(combined_cells_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=1)

        # Find connected components to count individual cells and get their properties
        # num_cells will include the background component (label 0)
        num_cells_found, labeled_cells, stats, centroids = cv2.connectedComponentsWithStats(combined_cells_mask_cleaned, 8, cv2.CV_32S)
        
        # Exclude background label (0) from analysis; stats[0] corresponds to background
        cell_areas_current = stats[1:, cv2.CC_STAT_AREA] 
        
        if len(cell_areas_current) == 0:
            print(f"  No cells detected in this predicted mask.")
            continue

        all_cell_areas.extend(cell_areas_current)
        total_cells_analyzed += len(cell_areas_current)

        # --- Polymegathism (Cell Shape Regularity - Hexagonality) ---
        # Find contours of each detected cell from the cleaned binary mask
        contours, _ = cv2.findContours(combined_cells_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hexagonal_count_current_image = 0
        for cnt in contours:
            # Skip very small contours that might be noise or artifacts
            if cv2.contourArea(cnt) < 10: # Adjust minimum area threshold if needed
                continue
            
            # Approximate the polygon using epsilon-poly approximation
            epsilon = 0.03 * cv2.arcLength(cnt, True) # 3% of arc length is a common starting point
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            num_vertices = len(approx)
            
            # Count if the approximated polygon is hexagonal (6 vertices)
            if num_vertices == 6:
                hexagonal_count_current_image += 1
        
        all_hexagonal_counts += hexagonal_count_current_image
        print(f"  {hexagonal_count_current_image} of {len(cell_areas_current)} detected components are approximately hexagonal.")

        # --- Visualization for this sample ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(current_original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(1, 3, 2)
        # Convert integer labels to displayable colors (e.g., for visualization)
        # 0: black, 1: blue, 2: green
        display_gt_mask = np.zeros((*current_ground_truth_mask.shape, 3), dtype=np.uint8)
        display_gt_mask[current_ground_truth_mask == 0] = [0, 0, 0] # Background (Black)
        display_gt_mask[current_ground_truth_mask == 1] = [0, 0, 255] # Border (Blue)
        display_gt_mask[current_ground_truth_mask == 2] = [0, 255, 0] # Interior (Green)
        plt.imshow(display_gt_mask)
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted Mask
        plt.subplot(1, 3, 3)
        display_pred_mask = np.zeros((*current_predicted_mask.shape, 3), dtype=np.uint8)
        display_pred_mask[current_predicted_mask == 0] = [0, 0, 0] # Background (Black)
        display_pred_mask[current_predicted_mask == 1] = [0, 0, 255] # Border (Blue)
        display_pred_mask[current_predicted_mask == 2] = [0, 255, 0] # Interior (Green)
        plt.imshow(display_pred_mask)
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.suptitle(f"Sample {i+1}: {current_filename} - Original vs. GT vs. Predicted Segmentation", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_plots_dir:
            plot_filename = os.path.splitext(current_filename)[0] + "_segmentation_comparison.png"
            plt.savefig(os.path.join(save_plots_dir, plot_filename))
            plt.close(fig) # Close the figure to free up memory
        else:
            plt.show()

    # --- Report Overall Morphological Metrics ---
    if total_cells_analyzed > 0:
        # Pleomorphism (Cell Size Variation) - Coefficient of Variation (CoV)
        mean_area = np.mean(all_cell_areas)
        std_area = np.std(all_cell_areas)
        cov_area = (std_area / mean_area) * 100 if mean_area > 0 else 0

        print("\n--- Morphological Analysis Results ---")
        print(f"Pleomorphism (Cell Size Variation):")
        print(f"  Coefficient of Variation (CoV) of Cell Areas: {cov_area:.2f}%")
        print(f"  (Mean Area: {mean_area:.2f} pixels, Std Dev Area: {std_area:.2f} pixels)")

        # Polymegathism (Cell Shape Regularity) - Percentage of Hexagonal Cells
        percentage_hexagonal = (all_hexagonal_counts / total_cells_analyzed) * 100
        print(f"\nPolymegathism (Cell Shape Regularity):")
        print(f"  Percentage of Hexagonal Cells: {percentage_hexagonal:.2f}%")
        print(f"  (Total cells analyzed for shape: {total_cells_analyzed}, Hexagonal cells: {all_hexagonal_counts})")
    else:
        print("\nNo cells were analyzed for overall morphological metrics.")


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Define your dataset paths ---
    # Path to the original images (raw TIFFs)
    original_images_directory = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/dataset'
    
    # Path to the ground-truth mask images (labelImages folder)
    masks_directory_for_segmentation = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/labelImages'

    # --- Step 1: Load and Preprocess Data for U-Net Training ---
    X, Y, filenames = load_segmentation_data(original_images_directory, masks_directory_for_segmentation, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)

    if X.size == 0 or Y.size == 0:
        print("No segmentation data loaded. Exiting.")
        exit()

    # Split data into training, validation, and test sets
    # Aim for approx 80% train, 10% validation, 10% test
    # We split filenames as well to reconstruct them for analysis
    X_train_val, X_test, Y_train_val, Y_test, filenames_train_val, filenames_test = train_test_split(
        X, Y, filenames, test_size=0.1, random_state=42
    )
    X_train, X_val, Y_train, Y_val, filenames_train, filenames_val = train_test_split(
        X_train_val, Y_train_val, filenames_train_val, test_size=(0.1/0.9), random_state=42
    )

    print(f"\nDataset Split for Segmentation:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input Image Shape: {X_train.shape[1:]}")
    print(f"  Masks (one-hot) Shape: {Y_train.shape[1:]}")

    # --- Step 2: Build and Compile the U-Net Model ---
    model = build_unet_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), num_classes=NUM_CLASSES)
    model.summary()

    # --- Step 3: Train the U-Net Model ---
    print("\n--- Training the U-Net Segmentation Model ---")
    
    # Define callbacks for training
    model_checkpoint_path = 'best_unet_model.h5' # .h5 is a widely compatible format
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Load the best model weights for evaluation
    try:
        model.load_weights(model_checkpoint_path)
        print(f"Loaded best model weights from {model_checkpoint_path}")
    except Exception as e:
        print(f"Could not load best model weights. Using last epoch's weights. Error: {e}")

    # --- Step 4: Evaluate Pixel-level Performance on Test Set ---
    dice_scores, iou_scores = evaluate_segmentation_performance(model, X_test, Y_test, NUM_CLASSES)

    # --- Step 5: Morphological Analysis on Predicted Masks from Test Set ---
    
    # Generate predictions for the test set
    Y_pred_proba_test = model.predict(X_test)
    Y_pred_labels_test = np.argmax(Y_pred_proba_test, axis=-1) # Convert to integer labels (H, W)

    # Define the directory to save segmentation comparison plots
    segmentation_plots_save_dir = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/segmentation_comparison_plots'
    
    # Select a few samples for detailed morphological analysis and visualization
    num_samples_for_morphology_analysis = 5 # Increased to 5 for more examples
    
    post_process_and_analyze_morphology(
        Y_pred_labels_test[:num_samples_for_morphology_analysis],
        X_test[:num_samples_for_morphology_analysis],
        Y_test[:num_samples_for_morphology_analysis], # Pass one-hot encoded GT for display
        filenames_test[:num_samples_for_morphology_analysis], # Pass corresponding filenames
        num_samples_to_analyze=num_samples_for_morphology_analysis,
        save_plots_dir=segmentation_plots_save_dir # Pass the save directory
    )

    # --- Optional: Plot Training History (Loss and Accuracy) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Segmentation Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Segmentation Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nObjective 3 Completed. Review the segmentation metrics, morphological analysis, and training history.")
    print(f"Segmentation comparison plots saved to: {segmentation_plots_save_dir}")
    print("Further improvements can be achieved through hyperparameter tuning, data augmentation, and unfreezing/fine-tuning the U-Net encoder.")
