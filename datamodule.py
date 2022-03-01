

from typing import Tuple

from torch.utils.data import DataLoader

from config import Dataset
from loaders import (Cifar10Loader, FashionMnistLoader,
                             MnistLoader, UMISTFacesLoader)


class DataModule():
    """
     A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """
    
    def __init__(self, 
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 pin_memory:bool,
                 dataset:Dataset,
                 train_val_test_split_percentage:Tuple[float,float,float]=(0.7,0.3,0.0),
                 input_size=None
                 
                 
                 ):
        
        super().__init__()
        self.data_dir=data_dir
        self.data_dir = data_dir
        self.train_val_test_split_percentage = train_val_test_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_enum=dataset
        self.input_size=input_size
        self.get_dataset()
        

    def get_dataset(self):
     
        if self.dataset_enum in [Dataset.cifar_crop,Dataset.cifar_ref]:
            self.dataset=Cifar10Loader
            self.in_chans=3
            if self.input_size is None:
                self.input_size=None
            
        elif self.dataset_enum in [Dataset.fashionmnist_noref,Dataset.fashionmnist_ref]:
            self.dataset=FashionMnistLoader
            self.in_chans=1
            if self.input_size is None:
                self.input_size=32
                
        elif self.dataset_enum== Dataset.mnist784_ref or  self.dataset_enum==Dataset.mnist784_classifier:
            self.dataset=MnistLoader
            self.in_chans=1
            if self.input_size is None:
                self.input_size=32
        elif self.dataset_enum==Dataset.umistfaces_ref:
            self.dataset=UMISTFacesLoader
            
            self.in_chans=1#comprobar
            if self.input_size is None:
                self.input_size=None

        else:
            raise ("select appropiate dataset")
    def prepare_data(self):
        """Se necesita el csv que proporciona Nando"""
        
        pass
    
    def setup(self,stage=None):
        """Load data. """
        self.fulldataset = self.dataset(self.data_dir,self.input_size)

    def dataloader(self):
        return DataLoader(
            dataset=self.fulldataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

 