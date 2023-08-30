import os
import argparse
import numpy as np
from collections import defaultdict
from utils import augmentation, mixup, mixup_criterion, get_combined_df
from model import ResNet18
from dataset import fundusDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss as CE
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

join = os.path.join
device = "cuda" if torch.cuda.is_available() else "mps"
rootDir = os.path.dirname(os.getcwd()) # project parent directory

#################################### Setting parameters #####################################
parser = argparse.ArgumentParser(description = "myopic_maculopathy_classifier")
parser.add_argument("--dataDir", 
                    help = "path to folder containing training images & groundtruth labels",
                    type = str, 
                    default = join(rootDir, "data")) 
parser.add_argument("--weightDir", 
                    help = "path to folder in which trained weights are to be saved",
                    type = str, 
                    default = join(rootDir, "weights"))
parser.add_argument("--batch_size", 
                    help = "batch size; default is 20",
                    type = int, 
                    default = 20)
parser.add_argument("--num_workers", 
                    help = "number of workers for dataloader; default is 0",
                    type = int, 
                    default = 0)
parser.add_argument("--lr", 
                    help = "Adam optimiser's initial learning rate; default is 5e-5",
                    type = float, 
                    default = 5e-5)
parser.add_argument("--weight_decay", 
                    help = "Adam optimiser's weight decay; default is 5e-4",
                    type = float, 
                    default = 5e-4)
parser.add_argument("--betas1", 
                    help = "Adam optimiser's 1st beta coefficient; default is 0.9",
                    type = int, 
                    default = 0.9)
parser.add_argument("--betas2", 
                    help = "Adam optimiser's 2nd beta coefficient; default is 0.999",
                    type = int, 
                    default = 0.999)
parser.add_argument("--eps", 
                    help = "Adam optimiser's epsilon (for numerical stability); default is 1e-8",
                    type = float, 
                    default = 1e-8)
parser.add_argument("--label_smoothing", 
                    help = "label smoothing for cross entropy loss function; default is 0.1",
                    type = float, 
                    default = 0.1)
parser.add_argument("--num_epochs", 
                    help = "total number of epochs; default is 50",
                    type = int, 
                    default = 50)
parser.add_argument("--mixup_prob", 
                    help = "probability of applying mixup augmentation; default is 0.5",
                    type = float, 
                    default = 0.5)
parser.add_argument("--mixup_alpha", 
                    help = "larger values result in greater mix-up between images; default is 0.4",
                    type = float, 
                    default = 0.4)
parser.add_argument("--plotTrainLosses",
                    help = "plot train loss vs epoch; default is True",
                    type = bool, 
                    default = True)

args = parser.parse_args()

###################################### Training setup #######################################
# Read cleaned dataframe with groundtruth labels
# for images from both MMAC and PALM challenges.
combined_df = get_combined_df(args.dataDir)

# Initialise a pretrained ResNet-18 model.
mmNet = ResNet18(num_classes = 5, 
                 pretrained  = True)
mmNet.to(device)

# Initialise train dataset & dataloader.
combinedTrainDataset    = fundusDataset(dataDir  = args.dataDir,
                                        dataframe = combined_df)
combinedTrainDataloader = DataLoader(combinedTrainDataset, 
                                     batch_size   = args.batch_size, 
                                     shuffle      = True, 
                                     num_workers  = args.num_workers)

# Initialise Adam optimiser with tuned lr, weight_decay, betas & eps values.
optim = torch.optim.Adam(mmNet.parameters(),
                         lr           = args.lr,
                         weight_decay = args.weight_decay,
                         betas        = (args.betas1, args.betas2),
                         eps          = args.eps)

# Compute the weight for each myopic maculopathy category, i.e.
# larger weight for less frequent category (patchy & macular atrophy).
combinedClassWeights = compute_class_weight(class_weight = 'balanced', 
                                            classes      = np.unique(combined_df.grade), 
                                            y            = combined_df.grade)
combinedClassWeights = torch.tensor(np.float32(combinedClassWeights))

# Initialise a weighted cross entropy loss function, with label smoothing
combinedLossFn = CE(weight          = combinedClassWeights.to(device), 
                    label_smoothing = args.label_smoothing,
                    reduction       = 'mean')

# Initialise cosine annealing scheduler.
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs)

# Initialise regular augmentations (random rotation,
# horizontal flip & brightness/saturation jitter).
augment = augmentation(angles = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30])

# Progress bar.
epoch_iter = trange(0, args.num_epochs, desc="Epochs") 

# Create an empty default dictionary to store 
# training metrics.
metrics = defaultdict(list)


####################################### Training begins #######################################
for i in epoch_iter:
    mmNet.train()
    batch_iter = tqdm(iter(combinedTrainDataloader), desc="Batches") 
    
    # Placeholder for batch losses summed across a training epoch
    total_batch_train_loss = 0.
    
    for j, (image, GToneHot) in enumerate(batch_iter):
        
        GToneHot = GToneHot.to(device)
        
        ##################################################################
        #  Apply Mixup augmentation to the entire batch with a 0.5 prob. #
        ##################################################################
        p = np.random.rand()
        if p < args.mixup_prob:
            image, GToneHot = mixup(image, GToneHot, alpha=args.mixup_alpha)
        ##################################################################
        
        # Apply regular augmentations. Input images need to be reshaped 
        # from [B,H,W,C] into [B,C,H,W] as expected by the "augment" 
        # function. Note that "augment" is applied to one image at a time
        # as opposed to applying it similarly to the entire batch at once.
        image = torch.tensor(np.uint8(image)).moveaxis(3, 1)
        image = transforms.Lambda(lambda image: torch.stack([augment(x) for x in image]))(image)
        
        # Normalise input image to [0-1] range
        image = image/255
        image = image.to(device, torch.float)
        
        optim.zero_grad()
        
        # Predict class probability.
        predicted_prob = mmNet(image)
            
        ##################################################################
        #    Apply Mixup criterion if mixup augmentation was applied.    #
        ##################################################################    
        if p < args.mixup_prob:
            ceLoss = mixup_criterion(combinedLossFn, predicted_prob, GToneHot)
        else:
            ceLoss = combinedLossFn(predicted_prob, GToneHot)
        ##################################################################
        
        # Backprop.
        ceLoss.backward()
        optim.step()
        
        # Keep adding training losses in this epoch. 
        total_batch_train_loss += ceLoss.item() 
    
    # Update learning rate.
    lr_scheduler.step()
    
    # Compute and save epoch loss.
    train_epoch_loss = total_batch_train_loss / (j+1)
    metrics["train_epoch_loss"].append(train_epoch_loss)
    print(train_epoch_loss)
    
    # Save weights after each epoch
    torch.save(mmNet.state_dict(), join(args.weightDir, str(i+1) + '_epoch_weights.pth'))
    
    # Plot train loss vs epoch
    if args.plotTrainLosses:
        plt.plot(metrics["train_epoch_loss"]) 



