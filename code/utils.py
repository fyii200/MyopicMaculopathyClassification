import os
import numpy as np
import torch
from torchvision.transforms import ColorJitter, RandomHorizontalFlip
from torchvision.transforms.functional import rotate
import pandas as pd
from PIL import Image, ImageFilter
join = os.path.join

################################################################################################
#################################### Regular augmentations #####################################
################################################################################################
class augmentation:
    def __init__(self, angles):
        """
        Args:
              angles (list): List of angles to be randomly sampled from.
        """
            
        self.angles = angles

    def __call__(self, input_image):
        """
        Args:
              input_image (3D or 4D tensor): Expect to have [...,C,H,W] shape.
        
        Out: 
              output_image (3D or 4D tensor): Augmented image (3D tensor) or images (4D tensor).
        """
        
        # Initialise random brightness & saturation jitter;
        # jitter factor randomly chosen from 0.5 to 1.5
        cj = ColorJitter(brightness=0.5, saturation=0.5)
        
        # Initialise random horizontal flip
        Hflip = RandomHorizontalFlip(p=0.5)
        
        # Randomly sample an angle from a predefined list of angles
        unif = torch.ones(len(self.angles))
        idx = unif.multinomial(1)
        angle = self.angles[idx]
        
        # Random rotation, followed by horizontal flip and brightness/saturation jitter
        output_image = cj(Hflip(rotate(input_image, angle)))
        
        return output_image

    
################################################################################################
##################################### Mixup augmentation #######################################
################################################################################################
def mixup(image, GToneHot, alpha):
    """
    Args:
          image (4D tensor)    : Expect to have [B,C,H,W] shape.
          GToneHot (2D tensor) : One-hot encoded label; expect to have [B,num_classes] shape 
          alpha (float)        : factor that controls the extent of mixup by influencing the 
                                 beta distribution from which the lambda value ("lam") is sampled. 
                                 lambda ranges from 0 to 1, where 0.5 indicates a 50:50 mixup between
                                 the first image and the second image. Larger alpha values are more 
                                 likely to yield lambda values that are closer to 0.5, giving rise to
                                 stronger regularisation. Smaller alpha values, on the other hand, are
                                 more likely to yield lambda values that are closer to either 0 or 1, 
                                 giving rise to little mixup between images.
    Out:
          new_image (4D tensor): Batch of new (composite) images; expect to have [B,C,H,W] shape.
          new_GToneHot (list)  : Contains labels in their original order, labels in their shuffled
                                 order and the randomly sampled lambda value.   
    """
    
    # Randomly shuffle the order of images and their 
    # corresponding labels in the current training batch
    indices = torch.randperm(image.size(0))
    shuffled_image = image[indices]
    shuffled_GToneHot = GToneHot[indices]          
    
    # Randomly sample a lambda ("lam") value from 
    # the beta distribution defined by alpha
    lam = np.random.beta(alpha, alpha)
    
    # New (composite) images are created by mixing images 
    # in their original order with images in their
    # shuffled order, i.e. two images per composite image
    new_image = image * lam + shuffled_image * (1 - lam)
    
    # Labels and lambda
    new_GToneHot = [GToneHot, shuffled_GToneHot, lam]
    
    return new_image, new_GToneHot


################################################################################################
######################################### Mixup loss ###########################################
################################################################################################
def mixup_criterion(criterion, pred_probs, targets):
    """
    Args:
          criterion (function)   : torch.nn loss function.
          pred_probs (2D tensor) : Predicted probability for each myopic maculopathy 
                                   category; expect to have [B, num_classes] shape.
          targets (list)         : "new_GToneHot" output by the "mixup" function above.
    Out:
          mixup_loss (1D tensor) : Losses with len(mixup_loss) = [B].
    """
    
    # Unpack targets
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    
    # Compute mixup loss
    mixup_loss = lam * criterion(pred_probs, targets1) + (1 - lam) * criterion(pred_probs, targets2)
    
    return mixup_loss


################################################################################################
###################### Return cleaned dataframe with labels for training #######################
################################################################################################
def get_combined_df(dataDir):
    """
    Read the dataframe with groundtruth labels for images
    from both MMAC and PALM challenges. Note that the 
    combined dataframe contains instances corresponding
    to PALM images without (5-class) labels because the
    macula in these images were obscured due to various
    reasons (e.g. retinal detachment), precluding robust 
    grading without OCT. These instances need to be removed.
    Besides, the dataframe also contains 250 synthesised
    images using PALM images as background. These images
    also need to be removed, as the author found that they
    were of little value, if not counterproductive, for
    training, compared with using only (synthesised) images
    with MMAC images as background.
    
    Args:
          dataDir (str)           : Path to the directory containing csv file
                                    with groundtruth labels for MMAC & PALM images.
    Out:
          combined_df (pandas df) : Cleaned dataframe with only instances
                                    to be used for training.
    """

    # Read dataframe
    combined_df = pd.read_csv(join(dataDir, 
                                   'Groundtruths', 
                                   'combinedTrainingLabels.csv'))
    
    # Remove PALM images that were not annotated
    combined_df = combined_df.dropna(subset = 'grade')
    
    # Remove synthesised images with PALM as background.
    # PS: 250 images (125 each with patchy atrophy and 
    # macular atrophy) to be removed.
    remove_names = []
    for i in range(126,251):
        remove_names.append("syn_patchy_" + str(i) + ".png")
        remove_names.append("syn_MA_" + str(i) + ".png")
        combined_df = combined_df[~combined_df.image.isin(remove_names)]
        
    return combined_df


################################################################################################
###################################### Image synthesiser #######################################
################################################################################################

class imageSynthesiser:
    def __init__(self, dataDir, lesion, lesionDataset, lesionBankDir):
        """
        Initialise image synthesiser. A new lesion and background image
        are randomly sampled each time this is this called.
        
        Args:
              dataDir (str)       : Path to the directory containing MMAC & PALM images
                                    along with their groundtruth labels.
              lesion (str)        : Type of myopic maculopathy lesion to be synthesised. 
                                    "MA" for macular atrophy; "patchy" for patchy atrophy.
              lesionDataset (str) : Name of dataset from which pertinent lesion is sampled.
                                    Accept either "MMAC" or "PALM".
              lesionBankDir (str) : Path to the (parent) directory containing individual folders,
                                    where each folder contains lesion masks of a particular
                                    myopic maculopathy category from a particular dataset. 
        """
            
        self.dataDir = dataDir
        self.lesion = lesion
        
        # Read the dataset that stores the groundtruth labels for each MMAC & PALM image.
        self.combinedData = pd.read_csv(join(dataDir, 
                                             "Groundtruths",
                                             "combinedTrainingLabels.csv"))
        
        # Specify path to the lesion bank subdirectory, depending on the type of
        # myopic maculopathy lesion and dataset selected.
        if (lesion == "MA") & (lesionDataset == "MMAC"):
            self.lesionDir = join(lesionBankDir, "MA_masked_MMAC")
            
        elif (lesion == "patchy") & (lesionDataset == "MMAC"):
            self.lesionDir = join(lesionBankDir, "patchy_masked_MMAC") 
            
        elif (lesion == "MA") & (lesionDataset == "PALM"):
            self.lesionDir = join(lesionBankDir, "MA_masked_PALM") 
            
        elif (lesion == "patchy") & (lesionDataset == "PALM"):
            self.lesionDir = join(lesionBankDir, "patchy_masked_PALM") 
            
        else:
            raise Exception("lesion must be one of: 'MA' or 'patchy', while lesionDataset must be one of: 'MMAC' or 'PALM'")

        # List of names of images from which pertinent lesions are derived (i.e. lesion image).
        self.lesionNames = list(os.listdir(self.lesionDir))
        if ".DS_Store" in self.lesionNames: self.lesionNames.remove('.DS_Store')
        
        # List of background image names (must come from MMAC, with either diffuse or patchy atrophy.
        diffuse_patchy_ind = (self.combinedData.grade == 3) | (self.combinedData.grade == 2)
        self.combinedData = self.combinedData[diffuse_patchy_ind]
        self.BInames = []
        for name in list(self.combinedData.image):
            if name[0:4] == "mmac":
                self.BInames.append(name)
        
        # Randomly sample a background image name
        self.randomBIname = np.random.choice(self.BInames)
        # Randomly sample a lesion image name
        self.randomLesionName = np.random.choice(self.lesionNames)

    def synthesise(self, rotateAngle, x_offset, y_offset, blurFilterSize = 25):
        """
        Args:
              rotateAngle (int)           : Degree of rotation applied to the lesion mask.
              x_offset (int)              : Amount of horizontal offset applied to the lesion mask.
                                            Note: set to 0 if lesion of interest is macular atrophy.
              y_offset (int)              : Amount of vertical offset applied to the lesion mask.
                                            Note: set to 0 if lesion of interest is macular atrophy.
              blurFilterSize (int)        : Gaussian blur filter size applied to the lesion mask.
        
        Out: 
              randomBI (PIL image)        : Background (MMAC) image to which lesion mask is applied.
              randomLesionImg (PIL image) : Image from which pertinent lesion mask is derived.
              synthesised (PIL image)     : Synthesised image.
        """
        
        # Open the randomly sampled background image.
        randomBI = Image.open(join(self.dataDir, "Images", "training", self.randomBIname ))
        
        # Open the randomly sampled lesion image.
        randomLesionImg = Image.open(join(self.dataDir, "Images", "training", self.randomLesionName ))
        # Make sure the lesion image has similar size as the background image.
        randomLesionImg = randomLesionImg.resize(size=randomBI.size)
        # Rotate the lesion image.
        randomLesionImgRotated = randomLesionImg.rotate(rotateAngle)
        # Open the (binary) lesion mask.
        randomLesionMask = Image.open(join(self.lesionDir, self.randomLesionName) ).convert('L')
        # Make sure the lesion mask has similar size as the background image.
        randomLesionMask = randomLesionMask.resize(size=randomBI.size)
        # Smooth the boundary of the lesion with a Gaussian blur filter.
        randomLesionMask = randomLesionMask.filter(ImageFilter.GaussianBlur(np.random.randint(blurFilterSize)))
        # Rotate the lesion mask by the same amount as that applied to the lesion image.
        randomLesionMaskRotated = randomLesionMask.rotate(rotateAngle)
        
        # Get the x and y coordinates of the fovea from the dataframe.
        fov_x = int(self.combinedData[self.combinedData.image == self.randomBIname].fovea_x)
        fov_y = int(self.combinedData[self.combinedData.image == self.randomBIname].fovea_y)
        
        # Get the x and y coordinates corresponding to the centre of the image
        centroid = np.mean(np.argwhere(randomLesionMaskRotated), axis = 0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
        
        # Offset the lesion in relation to the background image by "x_offset"
        # and "y_offset" amount. Note that if the lesion of interest is macular
        # atrophy ("MA"), setting offset to 0 ensures that the lesion is centred
        # on the fovea, which is what we want.
        if self.lesion == "MA":
            x = fov_x - centroid_x + x_offset
            y = fov_y - centroid_y + y_offset
        elif self.lesion == "patchy":
            x = x_offset
            y = y_offset
        
        # Interpolation between the background image and the post-processed (extracted) lesion.
        synthesised = randomBI.copy()
        synthesised.paste(randomLesionImgRotated, (x, y), randomLesionMaskRotated)
        
        return randomBI, randomLesionImg, synthesised





        
        
