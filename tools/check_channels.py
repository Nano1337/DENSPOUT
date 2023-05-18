from multiprocessing import Pool, cpu_count
from PIL import Image
import os
import numpy as np

# Worker function
def check_channels(filename):
    try:
        # Open an image file
        with Image.open(filename) as img:
            # Convert image data to a numpy array
            img_array = np.array(img)

        # If the number of channels is not 3, return the filename
        if img_array.shape[2] != 3:
            print("outlier shape is ", img_array.shape, " filename is ", filename)
            return filename
    except Exception as e:
        print(f'Error processing image {filename}: {e}')

    return None

def find_outliers(folder_path):
    # Get a list of image files in the directory
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create a pool of workers
    with Pool(cpu_count()) as p:
        results = p.map(check_channels, image_files)

    # Filter out None results
    outliers = [r for r in results if r is not None]

    # if outliers:
    #     # print("Outliers found:")
    #     # for filename in outliers:
    #     #     print(filename)
    #     # print("total outliers found: ", len(outliers))
    # else:
    #     print("No outliers")

# Test the function
find_outliers('/home/yinh4/DENSPOUT/data/dsWLC/trainA')
