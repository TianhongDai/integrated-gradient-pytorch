import numpy as np
import torch
from torchvision import models
from torchvision.utils import save_image
import cv2
import torch.nn.functional as F
from utils import calculate_outputs_and_gradients, generate_entrie_images
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot as plt 
from lit_regressor import LitRegressor
from config import CONFIG, Dataset,TargetModel
from datamodule import DataModule
# from torchsummary import summary

# sys.path.append("/content/adversarial_project/integrated-gradient-pytorch") #to work in colab
directorio = os.path.join(Path.home(), "integrated-gradient-pytorch")
sys.path.append(directorio) #para poder acceder a los ficheros como librerías (recuerda añadir __init__.py)
os.chdir(directorio) #para fijar este directorio como el directorio de trabajo

# python main.py --cuda --model-type='inception' --img='prueba.jpg'

parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='example', help='the type of network')
parser.add_argument('--img', type=str, default='mnist784_ref', help='the images name')
#parser.add_argument('--img', type=str, default='01.jpg', help='the images name')

if __name__ == '__main__':
    args = parser.parse_args()
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # start to create models...
    if args.model_type == 'inception':
        model = models.inception_v3(pretrained=True)
    elif args.model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    elif args.model_type == 'example':
        config = CONFIG
        model=LitRegressor(
            experiment_name=config.experiment_name,
            lr=config.lr,
            optim=config.optim_name,
            in_chans=config.in_chans,
            features_out_layer1=config.features_out_layer1,
            features_out_layer2=config.features_out_layer2,
            features_out_layer3=config.features_out_layer3,
            tanh1=config.tanh1,
            tanh2=config.tanh2,
            dropout1=config.dropout1,
            dropout2=config.dropout2,
            is_mlp_preconfig=config.is_mlp_preconfig,
            num_fold=0,
            num_repeat=0
                    )
        model = model.load_from_checkpoint(checkpoint_path='models/example.ckpt', in_chans = config.in_chans, experiment_name = config.experiment_name)
    
    model.eval()

    lista = [name for name, member in Dataset.__members__.items()]

    if args.img in lista:
        dataset_enum = Dataset[args.img]
        data_module = DataModule(os.path.join(config.path_data, dataset_enum.value), batch_size = config.batch_size, 
            num_workers = config.NUM_WORKERS, 
            pin_memory=True, 
            dataset = dataset_enum)
        data_module.setup()

        dataset = data_module.fulldataset
        img = dataset[1][0]
        imgnp = img.numpy()
        #img = np.transpose(img.numpy(), (1,2,0))
        print(img.shape)
        #save_image(dataset[1][0], "salida.png")
        print("Salida con éxito")

    else :
        # read the image
        img = cv2.imread('./examples/' + args.img)
        if args.model_type == 'inception':
            # the input image's size is different
            img = cv2.resize(img, (299, 299))
        img = img.astype(np.float32) 
        img = img[:, :, (2, 1, 0)]

        print(img.shape)

        print("Salida fracaso.")
    
    #sys.exit()
    if args.cuda:
        model.cuda()
        
      
    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, args.cuda, args.model_type != 'example')
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, imgnp, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
    img_gradient = visualize(gradients, imgnp, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

    sys.exit()


    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=50, num_random_trials=10, cuda=args.cuda)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                        img_integrated_gradient_overlay)
    cv2.imwrite('results/' + args.model_type + '/' + args.img, np.uint8(output_img))

    