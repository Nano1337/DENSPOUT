import os
import glob
import importlib
from dataset_utils.unaligned_dataset import UnalignedDataset
import utils

def get_dataset(dataset: str, args: dict):
    # Convert the dataset name to lower case for case-insensitive comparison
    dataset_lower = dataset.lower()

    # Look for the dataset folder in a case-insensitive way
    for name in os.listdir(args.dataroot):
        if name.lower() == dataset_lower:
            dataset = name  # Use the actual case of the found folder
            break
    else:
        raise FileNotFoundError(f"No folder named {dataset} found in {args.dataroot}")

    dataset_filename = os.path.join(args.dataroot, dataset)

    return UnalignedDataset(args, dataset_filename)