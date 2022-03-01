import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    resnet50="resnet50"
    densenet121="densenet121"
    vgg16="vgg16"
    alexnet="alexnet"
    googlenet="googlenet"
    tf_efficientnet_b0="tf_efficientnet_b0"
    tf_efficientnet_b4="tf_efficientnet_b4"
    tf_efficientnet_b7="tf_efficientnet_b7"
    
class Dataset (Enum):
    cifar_adversarial="cifar_crop_data_adversarial"
    cifar_crop="cifar-10.Class_Dffclt_Dscrmn_MeanACC.csv"
    cifar_replace="cifar-10-diff6replace.csv" # out of date
    cifar_ref="cifar-10.Diff6.RefClass.csv"# out of date
    fashionmnist_noref="Fashion-MNIST.Diff6.NoRefClass.csv"# out of date
    fashionmnist_ref="Fashion-MNIST.Class_Dffclt_Dscrmn_MeanACC.csv"
    mnist784_ref="mnist_784.Class_Dffclt_Dscrmn_MeanACC.csv"
    umistfaces_ref="UMIST_Faces.Class_Dffclt_Dscrmn_MeanACC.csv"
    mnist784_classifier="mnist_784V2.Clasification.csv"# out of date
    
    
class TargetModel(Enum):
    regresor_model=1
    classifier_model=2   
    
class Optim(Enum):
    adam=1
    sgd=2
    

@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.resnet50
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    
    target_model=TargetModel.regresor_model
    target_name:str=target_model.name
    #torch config
    batch_size:int = 1024
    dataset=Dataset.mnist784_ref
    dataset_name:str=dataset.name
    in_chans = 1
    precision_compute:int=32
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 0.01 #cambiar segun modelo y benchmark
    AUTO_LR :bool= False

    experiment_adversarial:bool=False #if True , then not exist Folds
    experiment_shift_dataset:bool=False
    experiment_with_blur:bool=False
    experiment_with_watermark:bool=True

    num_fold:int=0 #if 0 is not kfold train 
    repetitions:int=1
    
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    # IMG_SIZE:int=28
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")
    
    ##model
    features_out_layer1:int=1
    features_out_layer2:int=64
    features_out_layer3:int=64
    tanh1:bool=False
    tanh2:bool=False
    dropout1:float=0.2
    dropout2:float=0.3
    is_mlp_preconfig:bool=True
    
    ##data
    path_data:str=r"../adversarial_project/openml/data"
    
    gpu0:bool=False  
    gpu1:bool=True
    notes:str="final experiments"
    
    version:int=2
    

def create_config_dict(instance:CONFIG):
    return asdict(instance)


