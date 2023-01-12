from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose
from os.path import isfile, join
from os import listdir

def load_images(path_meta: str, path_images:str , type_dict: dict, cardinal_dict: dict) -> pd.DataFrame:
    """Load the HAM10000 images.

    Args:
        path_meta (str): Path to the metadata.
        path_images (str): Path to the images.
        type_dict (dict): Dictionary with the type of the disease.
        cardinal_dict (dict): Dictionary with the cardinality of the disease.

    Returns:
        pd.DataFrame: DataFrame with the metadata of the images.
    """
    
    meta = pd.read_csv(path_meta)[["image_id", "dx"]]
    images = [f for f in listdir(path_images) if isfile(join(path_images, f))]
    
    meta["image_id"] = meta['image_id'].map(lambda x: x + ".jpg")
    meta = meta[meta['image_id'].isin(images)]

    meta['image_id'] = meta['image_id'].map(lambda x: path_images + x)
    meta['type'] = meta['dx'].map(lambda x: type_dict[x])
    meta['label'] = meta['type'].map(lambda x: cardinal_dict[x])

    return meta.drop(columns=['dx'])


class HAM10000(Dataset):
    LESION_TYPE = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    TYPE_CARDINAL = {
        'Melanocytic nevi': 0,
        'dermatofibroma': 1,
        'Benign keratosis-like lesions': 2,
        'Basal cell carcinoma': 3,
        'Actinic keratoses': 4,
        'Vascular lesions': 5,
        'Dermatofibroma': 6
    }
    
    @staticmethod
    def load_from_df(df: pd.DataFrame, transform: Compose = None) -> Dataset:
        """Load the HAM10000 dataset from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with the metadata of the images.
            transform (Compose, optional): Transformations to apply. Defaults to None.

        Returns:
            Dataset: The HAM10000 dataset.
        """
        
        return HAM10000(df, transform)
    
    @staticmethod
    def load_from_file(root: str, train: bool = True, transform: Compose = None) -> Dataset:
        """Load the HAM10000 images from a path.

        Args:
            root (str): Path to the HAM10000 dataset.
            train (bool, optional): Get the training sample. Defaults to True.
            transform (Compose, optional): Transformation to apply. Defaults to None.

        Raises:
            FileNotFoundError: If the file are not found.

        Returns:
            Dataset: The HAM10000 dataset.
        """
        path_images = "/HAM10000_images_train/" if train else "/HAM10000_images_test/"
        path_metadata = "/HAM10000_metadata.csv"
        
        path_images = root + path_images
        path_metadata = root + path_metadata
        
        try:
            meta_df = load_images(path_metadata, path_images,
                                  HAM10000.LESION_TYPE, HAM10000.TYPE_CARDINAL)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path_metadata} or {path_images} not found.")
        
        return HAM10000(meta_df, transform)
    
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df.iloc[index, 0])
        
        if self.transform:
            X = self.transform(X)

        y = torch.tensor(self.df.iloc[index, 2], dtype=torch.long)

        return X, y
    
