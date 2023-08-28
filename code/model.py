import os
import cv2 as cv
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms.functional import rotate
join = os.path.join

class trainedModel:
    def __init__(self, checkpoint="bestModel.pth"):
        """
        Args:
              checkpoint (str): name of the file containing the trained weights (.pth).
        """
        
        self.checkpoint = checkpoint
        self.testHorFlip = transforms.RandomHorizontalFlip(p=1)
        self.testVerFlip = transforms.RandomVerticalFlip(p=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    def load(self, dir_path):
        """
        Args:
              dir_path (str): path to the weights.
        """
        
        # Load ResNet18 and its trained weights.
        self.model = ResNet18(num_classes=5)
        checkpoint_path = join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """
        Args:
              image (3D ndarray): input image of shape [H,W,C]; read using "cv2.imread".
        Out:
              pred_class (int)  : predicted myopic maculopathy category for the input image.
        """
        
        # Resize input image to 512 by 512 pixels and normalise to [0-1] range
        image = cv.resize(image, (512, 512))
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image/255
        image = image.to(self.device, torch.float)
        
        # Flip the image horizontally
        imageHorFlipped = self.testHorFlip(image)
        # Flip the image vertically
        imageVerFlipped = self.testVerFlip(image)
        
        with torch.no_grad(): 
            # Predict class probability from the original
            # image and 10 other variants of the same image.
            # Note: each of the following vectors has 5 float 
            # values, one for each class.
            score           = self.model(image)                   
            scoreHorFlipped = self.model(imageHorFlipped)         
            scoreVerFlipped = self.model(imageVerFlipped)
            scoreRotated1   = self.model(rotate(image, -5))
            scoreRotated2   = self.model(rotate(image, 5))
            scoreRotated3   = self.model(rotate(image, -8))
            scoreRotated4   = self.model(rotate(image, 8))
            scoreRotated5   = self.model(rotate(image, -12))
            scoreRotated6   = self.model(rotate(image, 12))
            scoreRotated7   = self.model(rotate(image, -15))
            scoreRotated8  = self.model(rotate(image, 15))
        
        # Take the average of the probability scores predicted
        # from the original image and its variants as the final
        # probability scores (i.e. one float value for each class).
        scoresSummed   = (score + 
                          scoreHorFlipped + 
                          scoreVerFlipped + 
                          scoreRotated1 + 
                          scoreRotated2 + 
                          scoreRotated3 + 
                          scoreRotated4 + 
                          scoreRotated5 + 
                          scoreRotated6 + 
                          scoreRotated7 + 
                          scoreRotated8)
        finalScores   = scoresSummed/11
        
        # Convert the final probability scores to an 
        # integer indicating the class membership.
        _, pred_class = torch.max(finalScores, 1)
        pred_class    = pred_class.detach().cpu()

        return pred_class
    
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        """
        Initialise a pre-trained ResNet18 model.
        """
        
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x   
    
    
