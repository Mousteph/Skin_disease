from os.path import isfile, join
from os import listdir
import pandas as pd

def get_images(path: str) -> list:
    """Return a list of all the images in a given path.

    Args:
        path (str): Path to the images.

    Returns:
        list: List of the paths to the images.
    """
    
    return [f for f in listdir(path) if isfile(join(path, f))]

def get_metadata(dataset: pd.DataFrame, images: list) -> pd.DataFrame:
    """Return the metadata of the images.

    Args:
        dataset (pd.DataFrame): Pandas DataFrame with the metadata.
        images (list[str]): List of the images.

    Returns:
        pd.DataFrame: DataFrame with the metadata of the images.
    """
    
    dataset["image_id"] = dataset['image_id'].map(lambda x: x + ".jpg")
    return dataset[dataset['image_id'].isin(images)]

def load_images(path_meta: str, path_images:str , type_dict: dict, cardinal_dict: dict) -> pd.DataFrame:
    """Load the images.

    Args:
        path_meta (str): Path to the metadata.
        path_images (str): Path to the images.
        type_dict (dict): Dictionary with the type of the disease.
        cardinal_dict (dict): Dictionary with the cardinality of the disease.

    Returns:
        pd.DataFrame: DataFrame with the metadata of the images.
    """
    
    meta = pd.read_csv(path_meta)[["image_id", "dx"]]
    images = get_images(path_images)
    meta = get_metadata(meta, images)

    meta['image_id'] = meta['image_id'].map(lambda x: path_images + x)
    meta['type'] = meta['dx'].map(lambda x: type_dict[x])
    meta['label'] = meta['type'].map(lambda x: cardinal_dict[x])

    return meta.drop(columns=['dx'])