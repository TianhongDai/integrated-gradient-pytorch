
from typing import Optional
import sys
import timm
import torch
import torch.nn as nn
from timm.models.factory import create_model
from torch.nn import functional as F
from torch.nn.modules import linear
from torchvision import models

from config import CONFIG, ModelsAvailable
from lit_system import LitSystem


class LitRegressor(LitSystem):
    
    def __init__(self,
                 experiment_name:str,
                lr:float,
                optim: str,
                in_chans:int,
                features_out_layer1:Optional[int]=None,
                features_out_layer2:Optional[int]=None,
                features_out_layer3:Optional[int]=None,
                tanh1:Optional[bool]=None,
                tanh2:Optional[bool]=None,
                dropout1:Optional[float]=None,
                dropout2:Optional[float]=None,
                is_mlp_preconfig:Optional[bool]=None,
                num_fold:Optional[int]=None,
                num_repeat:Optional[int]=None
                 ):
        
        super().__init__( lr, optim=optim,is_regresor=True)
        
        self.generate_model(experiment_name,
                            in_chans,
                                       features_out_layer1,
                                       features_out_layer2,
                                       features_out_layer3,
                                       tanh1,
                                       tanh2,
                                       dropout1,
                                       dropout2,
                                       is_mlp_preconfig
                                       )
        self.criterion=F.smooth_l1_loss #cambio de loss function
        self.num_fold=num_fold
        self.num_repeat=num_repeat
    
    def forward(self,x):
        return self.step(x)
    
    def step(self,x):

        y = self.model(x)

#        token_mean=self.token_mean.expand(x.shape[0],-1)
#        x=torch.cat((x,token_mean),dim=1)
#        y=self.regressor(x)
        y=torch.clamp(y,min=-6,max=+6)
        return y
    
    def training_step(self, batch,batch_idx):
        if len(batch)==3:
            x,targets,index=batch
        elif len(batch)==4:
            x,targets,index,labels=batch
        preds=self.step(x)
        loss=self.criterion(preds,targets)
        preds=torch.squeeze(preds,1)
        targets=torch.squeeze(targets,1)
        metric_value=self.train_metrics_base(preds,targets)
        data_dict={"loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        if len(batch)==3:
            x,targets,index=batch
        elif len(batch)==4:
            x,targets,index,labels=batch
        preds=self.step(x)
        loss=self.criterion(preds,targets)
        preds=torch.squeeze(preds,1)
        targets=torch.squeeze(targets,1)
        metric_value=self.valid_metrics_base(preds,targets)
        data_dict={"val_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
    def test_step(self, batch,batch_idx):
        if len(batch)==3:
            x,targets,index=batch
        elif len(batch)==4:
            x,targets,index,labels=batch
        preds=self.step(x)
        loss=self.criterion(preds,targets)
        preds=torch.squeeze(preds,1)
        targets=torch.squeeze(targets,1)
        metric_value=self.test_metrics_base(preds,targets)
        data_dict={"test_loss":loss,**metric_value}
        self.insert_each_metric_value_into_dict(data_dict,prefix="")
    
    def generate_model(self,
                        
                        experiment_name:str,
                        in_chans:int,
                        features_out_layer1:Optional[int]=None,
                        features_out_layer2:Optional[int]=None,
                        features_out_layer3:Optional[int]=None,
                        tanh1:Optional[bool]=None,
                        tanh2:Optional[bool]=None,
                        dropout1:Optional[float]=None,
                        dropout2:Optional[float]=None,
                        is_mlp_preconfig:Optional[bool]=None
                        ):
        
        if isinstance(experiment_name,str):
            model_enum=ModelsAvailable[experiment_name.lower()]
        if model_enum.value in timm.list_models(pretrained=True)  :
            extras=dict(in_chans=in_chans)
            self.model=timm.create_model(
                                        model_enum.value,
                                        pretrained=True,
                                        **extras
                                        )
            linear_sizes = [self.model.get_classifier().in_features]
        else:
            print("Modelo preentrenado no existente\n")
            sys.exit()
        
#        model_enum==ModelsAvailable.alexnet:
#            self.model=AlexNet(in_chans=in_chans)            
#        elif model_enum==ModelsAvailable.googlenet:
#            self.model=GoogleNet(in_chans=in_chans)
            
        # 
#        dim_parameter_token=2
#        if CONFIG.only_train_head:
#            for param in self.model.parameters():
#                param.requires_grad=False
#        self.token_mean=nn.Parameter(torch.zeros(dim_parameter_token))
#        
#        if model_enum==ModelsAvailable.resnet50:
#            linear_sizes = [self.model.fc.out_features+dim_parameter_token]
            # self.aditional_token=nn.Parameter(torch.zeros())
#        elif model_enum==ModelsAvailable.densenet121:
#            linear_sizes=[self.model.classifier.out_features+dim_parameter_token]
            # self.aditional_token=nn.Parameter(torch.zeros())
#        elif model_enum==ModelsAvailable.vgg16:
#            linear_sizes=[self.model.head.fc.out_features+dim_parameter_token]
        
#        elif model_enum==ModelsAvailable.alexnet:
#            linear_sizes=[256*3*3+dim_parameter_token]
        
#        elif model_enum==ModelsAvailable.googlenet:
#            linear_sizes=[1024+dim_parameter_token]
        
        
        if features_out_layer3:
            linear_sizes.append(features_out_layer3)
        
        if features_out_layer2:
            linear_sizes.append(features_out_layer2)
        if features_out_layer1:   
            linear_sizes.append(features_out_layer1)


        #if is_mlp_preconfig:
        #    self.regressor=Mlp(linear_sizes[0],linear_sizes[1])
        #else:  
        #    linear_layers = [nn.Linear(in_f, out_f,) 
        #               for in_f, out_f in zip(linear_sizes, linear_sizes[1:])]
        #    if tanh1:
        #        linear_layers.insert(0,nn.Tanh())
        #    if dropout1:
        #        linear_layers.insert(0,nn.Dropout(0.25))
        #    if tanh2:
        #        linear_layers.insert(-2,nn.Tanh())
        #    if dropout2:
        #        linear_layers.insert(-2,nn.Dropout(0.25))
        #    self.regressor=nn.Sequential(*linear_layers)

        #OJO CON LAS CABECERAS, DEPENDEN DEL MODELO
        #self.model.fc = self.regressor
        
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(linear_sizes[0]),
            nn.Linear(in_features=linear_sizes[0], out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.1), he quitado los bias = false de los nn.Linear
            nn.Linear(in_features=128, out_features=1)
        )

        # if model_enum==ModelsAvailable.resnet50:
        #     self.model.fc=self.regressor
        #     pass
        # elif model_enum==ModelsAvailable.densenet121:
        #     self.model.classifier=self.regressor
        #     pass

def swish(x):
    return x * torch.sigmoid(x)
ACT2FN={"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
class Mlp(nn.Module):
    def __init__(self, in_dim,hidden_dim,out_dim=1):
        super(Mlp, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
