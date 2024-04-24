## Background ([paper](https://link.springer.com/chapter/10.1007/978-3-031-54857-4_8))

Code for the work described in "A Clinically Guided Approach for Training Deep Neural Networks for Myopic Maculopathy Classification", which was carried out as part of the [2023 MICCAI Myopic Maculopathy Analysis Challenge](https://codalab.lisn.upsaclay.fr/competitions/12441). 

The task was to classify myopic maculopathy (MM) from colour fundus photographs based on the widely adopted META-PM framework, under which MM is categorised into normal (category 0), tessellated fundus (category 1), diffuse chorioretinal atrophy (category 2), patchy chorioretinal atrophy (category 3) and macular atrophy (category 4). Paper to appear in 2023 MICCAI Challenge Proceedings. 

The final model ranked first and fifth in the validation phase and test phase, respectively. The method employed (image synthesis -> regular + mix-up augmentations -> test-time augmentation) is summarised below:

![plot](fig/method_pipeline.jpg)
***PS: ResNet-18 diagram courtesy of [Ramzan et al.](https://link.springer.com/article/10.1007/s10916-019-1475-2)***


## Image synthesis pipeline
* Tutorial demonstrating the image synthesis pipeline can be found [here](code/imageSynthesis.ipynb).

## Training
* Train script can be found [here](code/train.py).
* Quickstart (in Jupyter notebook):
```
# Insert path to directory containing images and csv file with ground truth labels (default to "~/data") to
# the empty space marked by *. Note that the data directory "dataDir" is expected to have a subdirectory with
# the following relative path "Images/training" (where training images are stored) and another subdirectory
# "Groundtruths" where the csv file with ground truth labels is found ("combinedTrainingLabels.csv").

%run train --dataDir = *

```
#### Some info about data/Groundtruths/combinedTrainingLabels.csv
* csv file with ground truth labels for images from the MMAC dataset (row 1 to 1143), [PALM dataset](https://ieee-dataport.org/documents/palm-pathologic-myopia-challenge) (row 1144 to 2343), synthesised images (row 2344 to 2843). Note that there are 500 synthesised images in the csv file as opposed to the number reported in the manuscript (N=250) because half of the synthesised images were excluded from training (see the "get_combined_df" function in [utils.py](code/utils.py)), as they were created using PALM fundus images as background (but only synthesised images with ***MMAC*** images as background were of interest in this work). 

# Inference
The code snippet below demonstrates how to apply the trained model at inference:
```
from code.model import trainedModel
import cv2 as cv
import os
join = os.path.join

rootDir = os.path.dirname(os.getcwd()) 
model = trainedModel(checkpoint = "bestModel.pth")
model.load(dir_path = join(rootDir, "weights"))

### Please specify...
imageNames = [list of image names to be tested]
dataDir = [path to test image directory]

preds = []
for name in imageNames:
   image = cv.imread(join(dataDir, name))
   preds.append(int(model.predict(image)))
```

### If you use any part of this work, please cite
```
Yii, F. (2024). A Clinically Guided Approach for Training Deep Neural Networks for Myopic Maculopathy Classification. In: Sheng, B., Chen, H., Wong, T.Y. (eds) Myopic Maculopathy Analysis. MICCAI 2023. Lecture Notes in Computer Science, vol 14563. Springer, Cham. https://doi.org/10.1007/978-3-031-54857-4_8
```



