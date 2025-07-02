import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Added for data splitting

def generate_and_save_binary_masks(original_images_dir, output_labels_dir, visualize_conversion=False):
    """
    Generates binary masks from original grayscale images using adaptive thresholding
    and saves them to a specified output directory.
    The resulting masks will have white pixels (255) for foreground (cells)
    and black pixels (0) for background and borders, based on intensity.

    Args:
        original_images_dir (str): Path to the directory containing original grayscale images (.tif, .jpg).
        output_labels_dir (str): Path to the directory where the binary masks will be saved.
                                 This should be your 'labels' folder.
        visualize_conversion (bool): If True, displays the original and generated binary mask
                                     for each image. Close the plot to proceed to the next.
    """
    print(f"--- Starting generation of binary masks from: {original_images_dir} ---")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"Binary masks will be saved to: {output_labels_dir}")

    # Get all original image file paths (supporting .tif and .jpg)
    image_paths = sorted(glob(os.path.join(original_images_dir, '*.tif')))
    image_paths.extend(sorted(glob(os.path.join(original_images_dir, '*.jpg'))))
    image_paths = sorted(list(set(image_paths))) # Remove duplicates and sort again

    if not image_paths:
        print(f"No image files found in '{original_images_dir}' with .tif or .jpg extensions. Please check the path and file extensions.")
        return

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        # Save generated masks as .tif for consistency, regardless of original extension
        output_mask_path = os.path.join(output_labels_dir, os.path.splitext(img_filename)[0] + '.tif') 

        # Read the original image in grayscale
        img_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_grayscale is None:
            print(f"Warning: Could not load original image {img_path}. Skipping.")
            continue

        # --- Adaptive Thresholding for Binarization ---
        # Apply Gaussian blur to reduce noise and help with thresholding
        blurred_img = cv2.GaussianBlur(img_grayscale, (5, 5), 0)

        # Apply adaptive thresholding to convert to binary.
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Threshold value is a weighted sum of neighborhood values where weights are a Gaussian window.
        # 255: Maximum value to use with the THRESH_BINARY_INV type.
        # cv2.THRESH_BINARY_INV: Invert the binary output. This means pixels *below* the calculated threshold
        #                      will become 255 (white, representing cells), and pixels *above* will become 0 (black).
        #                      This is suitable if cells are generally darker than the background in your images.
        # Block size (e.g., 35): Size of a pixel's neighborhood for calculating threshold. Must be odd.
        # C (e.g., 5): Constant subtracted from the mean or weighted mean. Can be tuned.
        binary_mask = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 35, 5) # Tunable parameters

        # Optional: Morphological operations to clean up the mask
        # Opening (Erosion then Dilation) to remove small noise specks
        # Closing (Dilation then Erosion) to close small holes within cell regions
        kernel_morph = np.ones((3,3), np.uint8) # Small kernel
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_morph, iterations=1) # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=1) # Close small gaps

        # Save the generated binary mask
        cv2.imwrite(output_mask_path, binary_mask)
        print(f"Generated and saved: {img_filename} -> {os.path.basename(output_mask_path)}")

        # Visualize conversion if requested
        if visualize_conversion:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_grayscale, cmap='gray')
            plt.title(f'Original: {img_filename}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title(f'Generated Binary Mask')
            plt.axis('off')
            plt.suptitle(f"Generated Mask for {img_filename}", fontsize=14)
            plt.tight_layout()
            plt.show()

    print("--- Binary mask generation complete ---")


def calculate_cell_counts_from_binary_masks(mask_file_paths, visualize_intermediate=False, save_plots_dir=None):
    """
    Calculates the cell count for each mask image in the provided list of file paths.
    This version is designed for binary masks where white pixels (255) represent
    cell bodies and black pixels (0) represent borders and background.

    Args:
        mask_file_paths (list): A list of full paths to binary mask images.
        visualize_intermediate (bool): If True, displays intermediate images
                                       during processing for debugging.
        save_plots_dir (str, optional): If provided and visualize_intermediate is True,
                                        plots will be saved to this directory.

    Returns:
        dict: A dictionary where keys are image filenames and values are the
              corresponding cell counts. Returns an empty dictionary if no
              images are found or if the list of paths is empty.
    """
    cell_counts = {}
    print(f"--- Starting cell count calculation for {len(mask_file_paths)} binary masks ---")

    if not mask_file_paths:
        print("No mask file paths provided. Exiting cell count calculation.")
        return {}

    # Create save directory if specified and it doesn't exist
    if visualize_intermediate and save_plots_dir:
        os.makedirs(save_plots_dir, exist_ok=True)
        print(f"Plots will be saved to: {save_plots_dir}")
    elif visualize_intermediate and not save_plots_dir:
        print("Displaying plots interactively. Close each plot to proceed.")

    for mask_path in sorted(mask_file_paths): # Sorting ensures consistent order
        mask_filename = os.path.basename(mask_path)
        
        # Read the mask image in grayscale. Since it's binary, it will be 0 or 255.
        binary_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if binary_mask_raw is None:
            print(f"Warning: Could not load mask image {mask_path}. Skipping.")
            continue
        
        # Ensure the mask is strictly binary (0 or 255).
        # This handles cases where it might be loaded with other values if not perfectly binary.
        _, binary_mask = cv2.threshold(binary_mask_raw, 127, 255, cv2.THRESH_BINARY)
        # Here, 127 is a mid-point; any value > 127 becomes 255 (white), <= 127 becomes 0 (black).
        # This will correctly interpret your "white pixels for cell bodies" and "black for borders/background".

        # Apply connected components labeling directly to the binary mask.
        # This will count each distinct white region as a separate component (cell).
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)

        # The first label (label 0) is always reserved for the background.
        # Therefore, the number of foreground objects (cells) is num_labels - 1.
        cell_count = num_labels - 1
        
        # Ensure count is not negative if something unexpected happens
        if cell_count < 0:
            cell_count = 0 

        cell_counts[mask_filename] = cell_count

        # --- Optional: Visualize Intermediate Steps for Debugging / Saving ---
        if visualize_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
            axes = axes.flatten()

            titles = ['Raw Loaded Mask (Grayscale)', 'Final Binary Mask (Cells as White)']
            
            axes[0].imshow(binary_mask_raw, cmap='gray')
            axes[1].imshow(binary_mask, cmap='gray')
            
            for i, ax in enumerate(axes):
                ax.set_title(titles[i])
                ax.axis('off')
            
            plt.suptitle(f"Processing Steps for: {mask_filename} (Count: {cell_count})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if save_plots_dir:
                plot_filename = os.path.splitext(mask_filename)[0] + "_processed_binary_mask.png"
                plt.savefig(os.path.join(save_plots_dir, plot_filename))
                plt.close(fig)
            else:
                plt.show()

            print(f"Processed: {mask_filename}, Cells: {cell_count}") 

    print("--- Cell count calculation complete ---")
    return cell_counts

def visualize_cell_counts(cell_counts, title_suffix=""):
    """
    Generates and displays a histogram of cell counts.

    Args:
        cell_counts (dict): A dictionary of cell counts obtained.
        title_suffix (str): Suffix to add to the plot title (e.g., "Across Training Set").
    """
    if not cell_counts:
        print(f"\nCannot visualize: No cell counts data available to generate a histogram {title_suffix}.")
        return

    counts = list(cell_counts.values())

    plt.figure(figsize=(12, 7)) 
    
    min_count = min(counts)
    max_count = max(counts)
    
    if min_count == max_count:
        bins = [min_count - 0.5, min_count + 0.5] 
        plt.hist(counts, bins=bins, edgecolor='black', alpha=0.8, color='#607c8e')
        plt.xticks([min_count]) 
    elif max_count - min_count < 10:
        bins = np.arange(min_count, max_count + 2) - 0.5 
        plt.hist(counts, bins=bins, edgecolor='black', alpha=0.8, color='#607c8e')
        plt.xticks(range(min_count, max_count + 1))
    else:
        plt.hist(counts, bins=20, edgecolor='black', alpha=0.8, color='#607c8e')


    plt.title(f'Distribution of Cell Counts {title_suffix}', fontsize=16) 
    plt.xlabel('Number of Cells per Image', fontsize=12)
    plt.ylabel('Number of Masks', fontsize=12) 
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.7)
    
    avg_count = np.mean(counts)
    plt.axvline(avg_count, color='red', linestyle='dashed', linewidth=2, label=f'Average Count: {avg_count:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    print(f"\n--- Cell Count Statistics {title_suffix} ---")
    print(f"Total images analyzed: {len(counts)}")
    print(f"Minimum Cell Count: {np.min(counts)}")
    print(f"Maximum Cell Count: {np.max(counts)}")
    print(f"Average Cell Count: {np.mean(counts):.2f}")
    print(f"Median Cell Count: {np.median(counts)}")
    print(f"Standard Deviation of Cell Counts: {np.std(counts):.2f}")
    print("-----------------------------\n")


# --- Main execution block for Objective 1 ---
if __name__ == '__main__':
    # Define paths to your original images and where the new binary masks should be saved
    original_images_directory = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/dataset'
    output_labels_directory = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/labels'

    # 1. Generate and save binary masks from original images
    # Setting visualize_conversion to False to avoid too many pop-up windows during full execution.
    # You can set it to True for debugging the mask generation process if needed.
    generate_and_save_binary_masks(
        original_images_directory,
        output_labels_directory,
        visualize_conversion=False 
    )

    print("\nBinary mask generation complete. Now proceeding with cell count analysis and data splitting.")

    # Get all paths to the newly generated binary masks
    all_generated_mask_paths = sorted(glob(os.path.join(output_labels_directory, '*.tif')))

    if not all_generated_mask_paths:
        print(f"No binary mask files found in '{output_labels_directory}'. Cannot proceed with cell count analysis.")
        exit()

    # Split the generated binary mask paths into training and test sets
    # This split is for the *paths* to the masks, which will then be used to load data.
    train_mask_paths, test_mask_paths = train_test_split(
        all_generated_mask_paths, 
        test_size=0.2, # 20% for test, 80% for train
        random_state=42
    )

    print(f"\nDataset Split for Cell Count Analysis:")
    print(f"  Total generated binary mask files: {len(all_generated_mask_paths)}")
    print(f"  Training mask files for analysis: {len(train_mask_paths)}")
    print(f"  Test mask files (held out): {len(test_mask_paths)}")

    # Define the directory where plots for cell count analysis will be saved (optional).
    plots_save_directory_for_counts = 'C:/Users/VIJAY/Downloads/U-Net_Segmentation-Cornea_Cells-main/U-Net_Segmentation-Cornea_Cells-main/saved_plots_binary_labels'
    
    # 2. Calculate cell counts for the TRAINING SET of generated binary masks
    print("\n--- Running Cell Count Analysis on TRAINING SET of Generated Binary Masks ---")
    ground_truth_cell_counts_train = calculate_cell_counts_from_binary_masks(
        train_mask_paths, 
        visualize_intermediate=True, # Set to True to save/display plots for debugging the counting process
        save_plots_dir=plots_save_directory_for_counts 
    )

    # 3. Visualize the distribution of cell counts for the TRAINING SET
    visualize_cell_counts(ground_truth_cell_counts_train, title_suffix="Across Training Set (Generated Binary Masks)")

    print("\nObjective 1: Binary Mask Generation, Cell Count Analysis, and Data Splitting is complete.")
    print(f"Cell count analysis plots saved to: {plots_save_directory_for_counts}")
    print("The training and test sets of binary mask paths are now ready for subsequent objectives.")
