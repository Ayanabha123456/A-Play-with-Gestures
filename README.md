[<img src="https://img.shields.io/badge/Scikit Learn-ML-important.svg?logo=scikit-learn">](<LINK>)
[<img src="https://img.shields.io/badge/Tensorflow-Neural networks-important.svg?logo=tensorflow">](<LINK>)
[<img src="https://img.shields.io/badge/OpenCV-Image Processing-important.svg?logo=OpenCV">](<LINK>)
[<img src="https://img.shields.io/badge/Pandas-Data reading-important.svg?logo=pandas">](<LINK>)
[<img src="https://img.shields.io/badge/Numpy-Data Processing-important.svg?logo=numpy">](<LINK>)


<h1 align="center" style="font-size:60px;">A Play with Gestures</h1>
Sign Language Recognition and Production with Tensorflow and Scikit-learn developed as a final year project for Bachelors in Computer Science. <br></br>

# Technologies
<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="100">
<img src="https://www.gstatic.com/devrel-devsite/prod/vdbc400b97a86c8815ab6ee057e8dc91626aee8cf89b10f7d89037e5a33539f53/tensorflow/images/lockup.svg" width="100">
<img src="https://mediapipe.dev/images/mediapipe_small.png" width="100">
<img src="https://opencv.org/wp-content/uploads/2022/05/logo.png" width="40">
<img src="https://github.com/pandas-dev/pandas/blob/master/web/pandas/static/img/pandas_mark.svg" width="100">
<img src="https://numpy.org/images/logo.svg" width="50">

# Prerequisites
* Install [Anaconda](https://www.anaconda.com/products/distribution)
* Open Jupyter notebook

# Sign Language Recognition
It differentiates between various static hand gestures i.e. images, with the help of representative features with the core objective of maximizing the classification accuracy on the testing dataset.
## Datasets
<img src="https://www.researchgate.net/publication/362190853/figure/fig1/AS:1182452760494086@1658930115487/Datasets-ASL-alphabet-ASL-finger-spelling-dataset-Cambridge-dataset-NUS-I-and-II-ISL.ppm" style="display:block;margin:auto;" width="500">

## Feature Extraction
* Hand coordinates
* Convolutional features
* Convolutional features + finger angles
* Convolutional features on image with only hand edges
* Linear combination of [BRISK](https://ieeexplore.ieee.org/document/6126542) keypoints on edge image
* Convolutional features on [BRISK](https://ieeexplore.ieee.org/document/6126542) keypoint-infused image
<br></br>
Take a look at `Creating_datasets.ipynb` and `feature_extraction.ipynb` for the code.

## Dimensionality Reduction
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is applied only on the convolutional-related features since they are extensive in number.
## Classification
### Ensemble methods
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest) - takes aggregate result from a collection of decision trees
* [XGBoost](https://en.wikipedia.org/wiki/XGBoost) - each regression tree improves upon the result of the previous tree
<br></br>
Take a look at all `models-Ensemble methods` .ipynb files for the code.

### Neural Networks
* [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
<br></br>
Take a look at `models-ANN.ipynb` for the code.

* Hybrid ANN - takes original image and edge image in separate neural blocks, concatenates them and passes them through a final neural block
<br></br>
<img src="https://i.postimg.cc/m2YMckmF/hybrid-ANN.png" style="display:block;margin:auto;" width="300">
<br></br>
Take a look at all `models-hybrid ANN` .ipynb files for the code.

* [Transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
<br></br>
Take a look at `VGGNet.ipynb` for the code.

* [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
<br></br>
Take a look at `CNN-NUS I dataset.ipynb` and `CNN-NUS II dataset.ipynb` for the code.

# Sign Language Production
It generates interpretable sign images by training a generative model on the image dataset itself. The main aim is to generate as many image classes as possible all the while keeping a check on image quality.
## Datasets
<img src="https://i.postimg.cc/mZ1z1jbn/datasets-SLP.png" style="display:block;margin:auto;" width="500">

Since the above dataset comprise of videos, `Extracting video frames.ipynb` is used to get still images from the dataset.
## Latent Vector
This is the input to the generative model from which it gives images as output. It can either be random noise or noise generated from an encoder (`Autoencoder_generated_noise.ipynb`)
## [Generative Adversarial Network](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)
The following variants of the GAN are used in this project:

* Deep Convolutional GAN - uses convolutional layers in the generator and discriminator<br></br> 
Take a look at all `DCGAN` .ipynb files for the code.

* Wasserstein GAN - uses [Wasserstein loss](https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/) to output image scores for avoiding [mode collapse](https://machinelearning.wtf/terms/mode-collapse/) <br></br>
Take a look at all `WGAN` .ipynb files for the code.

* WGAN with Gradient Penalty - a modified WGAN that allows the model to learn complex functions for better mapping between real and fake images <br></br>
Take a look at all `WGAN_with_GP` .ipynb files for the code.

<i>Note:</i> 

Lower FID scores (`Fretchet Inception Distance.ipynb`) indicate better quality of images generated by the GAN models

# Publication
If you are referring this project for academic purposes, please cite the following paper:

Jana A, Krishnakumar SS. Sign Language Gesture Recognition with Convolutional-Type Features on Ensemble Classifiers and Hybrid Artificial Neural Network. <i>Applied Sciences.</i> 2022; 12(14):7303. https://doi.org/10.3390/app12147303
