import os
import re
import glob
from PIL import Image
import time

def identify_image_groups(image_folder):
    """
    Identify different image groups based on filename patterns.
    
    Args:
        image_folder (str): Path to the folder containing PNG images
        
    Returns:
        dict: Dictionary with group prefixes as keys and lists of image paths as values
    """
    # Get list of all PNG files in the folder
    all_images = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not all_images:
        print("[ERROR] No PNG images found in the folder.")
        return {}
    
    # Group images by their prefix
    image_groups = {}
    
    print(f"[INFO] Found {len(all_images)} PNG files in total")
    
    for image_path in all_images:
        filename = os.path.basename(image_path)
        # Extract the prefix (e.g., 'Jt_' or 'rho_')
        match = re.match(r'^([a-zA-Z]+_)', filename)
        
        if match:
            prefix = match.group(1)
            if prefix not in image_groups:
                image_groups[prefix] = []
            image_groups[prefix].append(image_path)
    
    # Print information about each group
    for prefix, images in image_groups.items():
        print(f"[INFO] Found {len(images)} images in group '{prefix}'")
    
    return image_groups

def create_gif_for_group(image_paths, output_filename, duration=100):
    """
    Create a GIF from a list of image paths.
    
    Args:
        image_paths (list): List of paths to PNG images
        output_filename (str): Name of the output GIF file
        duration (int): Duration for each frame in milliseconds
        
    Returns:
        bool: True if GIF was created successfully, False otherwise
    """
    if not image_paths:
        print(f"[ERROR] No images provided for creating GIF: {output_filename}")
        return False
    
    print(f"[INFO] Creating GIF from {len(image_paths)} images...")
    print(f"[INFO] First few files: {[os.path.basename(p) for p in image_paths[:3]]}")
    print(f"[INFO] Last few files: {[os.path.basename(p) for p in image_paths[-3:]]}")
    
    start_time = time.time()
    
    # Sort images based on the numerical value in their filenames
    def extract_number(filename):
        match = re.search(r'(\d+\.\d+)', os.path.basename(filename))
        if match:
            return float(match.group(1))
        return 0
    
    print("[INFO] Sorting images...")
    image_paths.sort(key=extract_number)
    
    # Sample the sorted order to verify it looks correct
    print(f"[INFO] First 3 sorted files: {[os.path.basename(p) for p in image_paths[:3]]}")
    print(f"[INFO] Last 3 sorted files: {[os.path.basename(p) for p in image_paths[-3:]]}")
    
    # Load all images
    print("[INFO] Loading images...")
    frames = []
    for i, image_path in enumerate(image_paths):
        try:
            if i % 20 == 0:  # Print progress every 20 frames
                print(f"[INFO] Loading image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            if i % 1 == 0:  # Skip every other image for faster loading
                frames.append(Image.open(image_path))
        except Exception as e:
            print(f"[ERROR] Failed to open {image_path}: {e}")
    
    if not frames:
        print("[ERROR] No frames were loaded successfully")
        return False
    
    # Save as GIF
    print(f"[INFO] Saving GIF with {len(frames)} frames to {output_filename}...")
    try:
        frames[0].save(
            output_filename,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0  # 0 means loop indefinitely
        )
        print(f"[SUCCESS] Created GIF with {len(frames)} frames: {output_filename}")
        print(f"[INFO] GIF creation took {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save GIF: {e}")
        return False

if __name__ == "__main__":
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Working directory: {current_dir}")
    
    # Duration of each frame in milliseconds (increased for slower animation)
    # Higher values make the animation slower
    duration = 600  # Changed from 100ms to 300ms
    print(f"[INFO] Setting frame duration to {duration}ms (higher = slower animation)")
    
    # Identify different image groups
    image_groups = identify_image_groups(current_dir)
    
    if not image_groups:
        print("[WARNING] No image groups found. Make sure PNG files exist with proper naming.")
    
    # Create a GIF for each group
    for prefix, images in image_groups.items():
        # Remove trailing underscore for the output filename
        group_name = prefix.rstrip('_')
        output_filename = os.path.join(current_dir, f"{group_name}_animation.gif")
        
        print(f"\n[INFO] Processing group '{prefix}' with {len(images)} images")
        print(f"[INFO] Output file will be: {output_filename}")
        
        if create_gif_for_group(images, output_filename, duration):
            print(f"[SUCCESS] {prefix} animation created successfully!")
        else:
            print(f"[ERROR] Failed to create animation for {prefix} group")
    
    print("\n[INFO] All GIF creation tasks completed!")