import os

def rename_images_in_directory(directory, prefix):
    """
    Rename all image files in a directory sequentially with the specified prefix.

    Parameters:
    directory (str): Path to the directory containing image files.
    prefix (str): Prefix for the new filenames (e.g., "RGB" or "thermal").
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Get list of files in the directory
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"No images found in {directory}")
        return
    
    # Sort files for consistent renaming
    files.sort()
    
    for idx, file in enumerate(files, start=1):
        new_name = f"{prefix}{idx:04d}.jpg"
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

# Base directory and subdirectories to process
base_dir = "flir" 
directories = {
    "images_rgb_train/data": "RGB",
    "images_rgb_val/data": "RGB",
    "images_thermal_train/data": "thermal",
    "images_thermal_val/data": "thermal",
}

# Rename images in each directory
for sub_dir, prefix in directories.items():
    target_dir = os.path.join(base_dir, sub_dir)
    print(f"Renaming files in: {target_dir} with prefix: {prefix}")
    rename_images_in_directory(target_dir, prefix)

