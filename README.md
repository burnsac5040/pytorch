## PyTorch Projects

### [Fruits 360 Dataset from Kaggle](https://www.kaggle.com/moltean/fruits)
#### [Convolutional Neural Network](fruits-360)
A model was built using PyTorch to classify images of 131 types of fruit.  This may not be the best code out there, as it was my first attempt at using PyTorch.

Early stopping was implemented by taking a class from the `pytorchtools` (`pip install pytorchtools`) module.  I was unable to use the class when importing the module, so it was taken from the source code and put into this project instead.

Graphs were made of the model's confidence of predictions as well as the ROC curve.  The model's saved state after being trained on the images is also in this repository as a [file](fruits-360/2-model-a2.pt).
