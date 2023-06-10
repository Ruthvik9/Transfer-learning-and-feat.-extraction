import os
import torch
import pickle
import cv2 as cv
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
    
    
    
class FeatureExtraction(nn.Module): # nn.Module is the base class for all neural networks.
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model # The model that will be used to extract the features.
        self.layer = self.model._modules.get('avgpool') # Extracting features from the avgpool layer.
                                                        # Feel free to change this to any other layer.
        
        self.fe = None  # Container vector for storing the extracted features
        self.layer.register_forward_hook(self.func) # Registering the forward hook.
        
    def func(self,layer,_,output):          # Function to be attached to the avgpool layer.
            print(f"Extracting from {layer}")
            self.fe = torch.zeros(output.shape)
            self.fe.copy_(output.data) # Save the output in the fe variable.


    def forward(self, x):                      # Forward method of the neural network.
# =============================================================================
#         for m in self.model.modules():        # To disable batch normalization so that the model gives
#             if isinstance(m, nn.BatchNorm2d):  # same results irrespective of the batch size.
#                 m.eval()
# =============================================================================
        self.model.eval()
        return self.model(x)                   # When the forward pass of avgpool happens, the hook is called.
    
class pacp(Dataset):
    """Custom dataset class for the pacp-dataset"""
    def __init__(self, fnames, root_dir, transform=True):
        """
        Args:
            fnames (list): List of all the file names.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fnames = fnames 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        print(idx)
        #Transforms to resize the image to 224x224 px
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.482, 0.458, 0.407], # If we have a grayscale image, we need to 
                                         std=[0.0039, 0.0039, 0.0039])  # specify mean and std for one channel only.
        to_tensor = transforms.ToTensor()
        
        img_name = os.path.join(self.root_dir,
                                self.fnames[idx] + ".jpg")
        image = cv.imread(img_name)
        
        if self.transform:
            image = normalize(scaler(to_tensor(image)))  # As described in the paper, the image is resized and then normalized.
        return image                                     # But as in the paper, we didn't centercrop since we may lose important information.


# transform = torch.nn.Sequential(
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)), # Resize images into 224 X 224 for ResNet.
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], # If we have a grayscale image, we need to 
#                                  std=[0.229, 0.224, 0.225]),  # specify mean and std for one channel only.
    
# )

if __name__ == "__main__":
    
    # Load the models pretrained on the imagenet dataset.
    # model_res50 = models.resnet50(pretrained = "imagenet") # A better version is used below.
    vgg16 = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_FEATURES)
    

    # Load file names from pickle file
    with open("fnames","rb") as f:
        fnames = pickle.load(f)
        
    # Creating a DataLoader to transform and load images of our custom dataset
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") # Defining the device (for using its GPU.)
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(sci_mode = False)
    
    
    training_set = pacp(fnames,"D:/ilinkfiles/ilink/Approved",transform = True)
    print("Length of dataset:",training_set.__len__())
    
    # Parameters for loading data
    params = {'batch_size': 16,
              'shuffle': False,
              'num_workers': 8}
    training_generator = DataLoader(training_set, **params)
    featext = FeatureExtraction(vgg16) # Using the vgg16 model to extract features.
    lists = []
    for local_batch in training_generator:
            # print(local_batch)
            # Transfer to GPU
            featext = featext.to(device) # Shifting the model to the GPU.
            local_batch = local_batch.to(device) # Shifting the data to the GPU.
            _ = featext(local_batch)
            # print(outputs)
            # outputs2 = model_res50(local_batch)
            # print(outputs2)
            local_batch.detach()      # Detaching the data from the GPU so as to not take up all memory.
            # print(featext.fe.shape)
            # print("Printing extracted feature:")
            # print(featext.fe)
            lists.append(featext.fe.flatten(start_dim = 1))
            torch.cuda.empty_cache()
            # print(featext.fe)
            
  
    res = torch.cat(lists,0).numpy()
    print("Output size:", res.shape)
    with open("vgg16fe","wb") as f:
        pickle.dump(res,f)
    # print(res)
# =============================================================================
#     with open("batch16","wb") as f:
#        pickle.dump(res.numpy(),f)
# =============================================================================
    # print(torch.cat(lists,0).numpy())


