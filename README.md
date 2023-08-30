## Domain knowledge-guided training

Code for the work described in "Domain knowledge-guided training of a deep neural network for myopic maculopathy classification", which was carried out as part of the [2023 MICCAI Myopic Maculopathy Analysis Challenge](https://codalab.lisn.upsaclay.fr/competitions/12441). 

The task was to classify myopic maculopathy (MM) from colour fundus photographs based on the widely adopted META-PM framework, under which MM is categorised into normal (category 0), tessellated fundus (category 1), diffuse chorioretinal atrophy (category 2), patchy chorioretinal atrophy (category 3) and macular atrophy (category 4). Paper to appear in 2023 MICCAI Challenge Proceedings. 

The final model ranked [first](https://codalab.lisn.upsaclay.fr/competitions/12441#results) and [sixth](https://codalab.lisn.upsaclay.fr/competitions/12441#results) in the validation phase and test phase, respectively. The method employed (image synthesis -> regular + mix-up augmentations -> test-time augmentation) is summarised below:

![plot](fig/method_pipeline.jpg)

# Image synthesis pipeline
* Tutorial demonstrating the image synthesis pipeline can be found [here](code/imageSynthesis.ipynb)

# Inference
The code snippet below demonstrates how to apply the trained model at inference:





