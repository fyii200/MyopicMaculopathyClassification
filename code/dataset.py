import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
join = os.path.join

class fundusDataset(Dataset):

    def __init__(self, dataDir, dataframe, num_classes=5):
        
        """
        Args:
             dataDir (str)                : Parent directory with the training data (images and 
                                            csv file with groundtruth labels).
             dataframe (2D tabular)       : pandas dataframe with image names and groundtruth 
                                            annotation (from both MMAC and PALM datasets).
             num_classes (int)            : number of classes (5 by default).
        Out:
             image (4D tensor)            : batch of images of shape [B,H,W,C]
             mmCategoryOneHot (2D tensor) : one-hot encoded labels of shape [B,num_classes]
        """
        self.dataDir     = dataDir
        self.dataframe   = dataframe
        self.num_classes = num_classes
        self.device      = "cuda" if torch.cuda.is_available() else "cpu" 

    def __len__(self):
        
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        
        # Read image
        img_name = join(self.dataDir,
                        "Images",
                        "training",
                        self.dataframe.iloc[idx, 0])
        image = cv.imread(img_name)
        
        # Resize image to 512 by 512 pixels
        image = cv.resize(image, (512, 512))          
        
        # groundtruth label (second column): 0 to 4 (normal to macular atrophy).
        mmCategory = self.dataframe.iloc[idx, 1]
        
        # One-hot encode groundtruth label
        mmCategory = torch.tensor(np.float32(mmCategory)) 
        mmCategoryOneHot = torch.zeros(self.num_classes)
        mmCategoryOneHot[np.uint8(mmCategory)] = 1

        return image, mmCategoryOneHot 
    
    

    
    
    
    
    
    
