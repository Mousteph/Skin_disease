from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose
from utils_load_images import load_images


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
    
