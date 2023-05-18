import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

def process_image(filename):
    try:
        # Open an image file
        with Image.open(filename) as img:
            # Convert the image to RGB, discarding any alpha channel
            img = img.convert('RGB')

            # Convert image data to a numpy array
            img_array = np.array(img)

        # Calculate min, max, mean and standard deviation for each channel
        mins = img_array.min(axis=(0, 1))
        maxs = img_array.max(axis=(0, 1))
        means = img_array.mean(axis=(0, 1))
        stds = img_array.std(axis=(0, 1))

        # Normalize image
        eps = 1e-7  # small constant to prevent division by zero
        img_array = (img_array - mins) / (maxs - mins + eps)

        # Calculate mean and std of normalized image for each channel
        norm_means = img_array.mean(axis=(0, 1))
        norm_stds = img_array.std(axis=(0, 1))

        return mins, maxs, means, stds, norm_means, norm_stds
    except Exception as e:
        print(f'Error processing image {filename}: {e}')
        return None


def get_image_stats(folder_path):
    # Get a list of image files in the directory
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create a pool of workers
    with Pool(cpu_count()) as p:
        results = p.map(process_image, image_files)

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("No results returned from worker function.")
        return None

    # Separate mins, maxs, means, stds, norm_means, norm_stds
    mins, maxs, means, stds, norm_means, norm_stds = zip(*results)

    # Calculate overall mean and standard deviation
    overall_mins = np.array(mins).min(axis=0)
    overall_maxs = np.array(maxs).max(axis=0)
    overall_means = np.array(means).mean(axis=0)
    overall_stds = np.array(stds).mean(axis=0)
    overall_norm_means = np.array(norm_means).mean(axis=0)
    overall_norm_stds = np.array(norm_stds).mean(axis=0)

    return overall_norm_means, overall_norm_stds


if __name__ == '__main__':
    folder = '/home/yinh4/DENSPOUT/data/dsWLC/trainB'    
    means, stds = get_image_stats(folder)
    print("means:", means)
    print("stds:", stds)