# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:22:43 2025

@author: VIJAY
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50V2 # For transfer learning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error # Added for RMSE calculation

# --- Constants for Image Processing and Model Training ---
IMG_SIZE = 224 # ResNet50V2 expects 224x224 input
IMG_CHANNELS = 3 # ResNet50V2 expects 3 channels (will convert grayscale to 3 identical channels)
BATCH_SIZE = 16 # Adjust based on your GPU memory
EPOCHS = 50     # Initial number of epochs; EarlyStopping will manage this

# --- Helper Function: Count Cells from a Binary Mask (NumPy Array) ---
def count_cells_from_binary_mask_array(binary_mask_array):
    """
    Calculates the cell count from a binary mask provided as a NumPy array.
    Assumes white pixels (255) represent cell bodies and black pixels (0) represent background/borders.

    Args:
        binary_mask_array (np.array): A 2D NumPy array representing the binary mask.

    Returns:
        int: The count of individual cells.
    """
    # Ensure the mask is strictly binary (0 or 255) before connected components
    _, binary_mask = cv2.threshold(binary_mask_array, 127, 255, cv2.THRESH_BINARY)

    # Apply connected components labeling
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    
    # The first label (label 0) is always reserved for the background.
    # Therefore, the number of foreground objects (cells) is num_labels - 1.
    cell_count = num_labels - 1 
    if cell_count < 0: cell_count = 0 # Ensure count is not negative

    return cell_count


# --- 1. Data Loading and Preprocessing for Regression ---
def load_data_for_regression(original_images_dir, img_size):
    """
    Loads and preprocesses original images and derives their ground-truth cell counts
    by generating binary masks on-the-fly and counting cells from them.

    Args:
        original_images_dir (str): Path to the directory containing original (raw) images.
        img_size (int): Desired width/height to resize images to.

    Returns:
        tuple: A tuple (X, Y, filenames) where X is a NumPy array of preprocessed images,
               Y is a NumPy array of corresponding cell counts, and filenames is a list.
    """
    X = [] # List to store preprocessed original images
    Y = [] # List to store corresponding cell counts
    filenames = [] # List to store original image filenames

    print(f"--- Loading images from: {original_images_dir} and deriving cell counts ---")

    # Get all original image paths (supporting .tif and .jpg)
    original_image_paths = sorted(glob(os.path.join(original_images_dir, '*.tif')))
    original_image_paths.extend(sorted(glob(os.path.join(original_images_dir, '*.jpg'))))
    original_image_paths = sorted(list(set(original_image_paths))) # Ensure unique and sorted

    if not original_image_paths:
        print("Error: No original image files found. Check path and file types.")
        return np.array([]), np.array([]), []

    for original_path in original_image_paths:
        filename = os.path.basename(original_path)
        
        # Load original image (grayscale)
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None:
            print(f"Warning: Could not load original image {original_path}. Skipping.")
            continue

        # --- On-the-fly Binary Mask Generation for Ground Truth ---
        # Apply Gaussian blur for noise reduction
        blurred_img = cv2.GaussianBlur(original_img, (5, 5), 0)

        # Apply adaptive thresholding to convert to binary.
        # THRESH_BINARY_INV is used assuming cells are darker and should become white (255).
        temp_binary_mask = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY_INV, 35, 5) # Tunable parameters

        # Optional: Morphological operations to clean up the mask
        kernel_morph = np.ones((3,3), np.uint8)
        temp_binary_mask = cv2.morphologyEx(temp_binary_mask, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        temp_binary_mask = cv2.morphologyEx(temp_binary_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=1)

        # Calculate cell count from the generated binary mask
        cell_count = count_cells_from_binary_mask_array(temp_binary_mask)

        # --- Preprocess Original Image for Model Input ---
        resized_original_img = cv2.resize(original_img, (img_size, img_size))
        normalized_original_img = resized_original_img / 255.0 # Normalize to [0, 1]

        # Expand grayscale to 3 channels for ResNet50V2 input
        original_img_3_channel = np.stack([normalized_original_img, normalized_original_img, normalized_original_img], axis=-1)

        X.append(original_img_3_channel)
        Y.append(cell_count)
        filenames.append(filename)

    print(f"--- Loaded {len(X)} image-count pairs for regression training ---")
    return np.array(X), np.array(Y), filenames


# --- 2. Build Regression Model Architecture (ResNet50V2 with Regression Head) ---
def build_resnet_regression_model(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)):
    """
    Builds a regression model using a pre-trained ResNet50V2 backbone.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: Compiled Keras regression model.
    """
    # Load ResNet50V2 base model pre-trained on ImageNet, excluding the top (classification) layer
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model so they are not updated during training
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom regression head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Flatten the feature maps
    x = Dense(256, activation='relu')(x) # First dense layer
    x = Dense(128, activation='relu')(x) # Second dense layer
    predictions = Dense(1, activation='linear')(x) # Output layer for regression (single continuous value)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    # Use Mean Squared Error (MSE) for loss and Mean Absolute Error (MAE) as a metric
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=MeanSquaredError(),
                  metrics=[MeanAbsoluteError()])

    return model

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # --- Define your dataset paths ---
    # Path to the original grayscale images, which will now also serve as the source
    # for deriving ground-truth cell counts on-the-fly.
    original_images_directory = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/dataset'
    
    # The 'colored_masks_directory' is no longer needed for ground truth in this objective.
    # It's kept as a commented line for context if you still have those files.
    # colored_masks_directory = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/labelImages'

    # --- Step 1: Load and Preprocess Data for Regression Training ---
    # The ground truth cell counts are now derived directly from the original images.
    X, Y, filenames = load_data_for_regression(original_images_directory, IMG_SIZE)

    if X.size == 0 or Y.size == 0:
        print("No data loaded for regression. Exiting.")
        exit()

    # --- Print cell count for each image in the loaded dataset ---
    print("\n--- Ground Truth Cell Count for Each Image in the Dataset ---")
    for i in range(len(filenames)):
        print(f"Image: {filenames[i]}, Ground Truth Cell Count: {Y[i]:.0f}")
    print("-----------------------------------------------------------\n")

    # Split data into training, validation, and test sets
    # Aim for approx 80% train, 10% validation, 10% test
    # We split filenames as well to reconstruct them for analysis
    X_train_val, X_test, Y_train_val, Y_test, filenames_train_val, filenames_test = train_test_split(
        X, Y, filenames, test_size=0.1, random_state=42 # 10% for test
    )
    X_train, X_val, Y_train, Y_val, filenames_train, filenames_val = train_test_split(
        X_train_val, Y_train_val, filenames_train_val, test_size=(0.1/0.9), random_state=42 # 10% of remaining 90% for validation
    )

    print(f"\nDataset Split for Regression:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input Image Shape: {X_train.shape[1:]}")
    print(f"  Output Label Shape (Cell Count): {Y_train.shape}") # Should be (num_samples,)

    # --- Step 2: Build and Compile the Regression Model ---
    model = build_resnet_regression_model(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    model.summary()

    # --- Step 3: Train the Regression Model ---
    print("\n--- Training the Regression Model ---")
    
    # Define callbacks for training
    model_checkpoint_path = 'trained_regression_model.h5'
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

    # --- Step 4: Evaluate the Model on Test Set ---
    print("\n--- Evaluating Regression Model on Test Set ---")
    loss, mae = model.evaluate(X_test, Y_test, verbose=0)
    rmse = np.sqrt(mean_squared_error(Y_test, model.predict(X_test))) # Calculate RMSE
    print(f"Test Loss (MSE): {loss:.4f}")
    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}") # Display RMSE

    # --- Optional: Predict and display some results from the test set ---
    print("\n--- Sample Predictions on Test Set ---")
    Y_pred_test = model.predict(X_test)
    for i in range(min(5, len(X_test))): # Display first 5 predictions
        print(f"Image: {filenames_test[i]}, True Count: {Y_test[i]:.0f}, Predicted Count: {Y_pred_test[i][0]:.0f}")

    # --- Optional: Plot Training History (Loss and MAE) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Regression Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Regression Model Mean Absolute Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- New: Scatter Plot of Predicted vs. True Values ---
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_test, Y_pred_test, alpha=0.6)
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', label='Ideal Prediction (True = Predicted)') # y=x line
    plt.title('Predicted vs. True Cell Counts on Test Set')
    plt.xlabel('True Cell Count')
    plt.ylabel('Predicted Cell Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("\nObjective 2: Automated Cell Count Estimation (Regression) is complete.")
    print(f"Trained model saved to: {model_checkpoint_path}")
