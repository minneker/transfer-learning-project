## Welcome to my final project page!

We set out to use deep transfer learning with popular convolutional neural network architectures to diagnose Parkinson's disease (PD) from images of handwritten spirals (i.e. classify healthy vs. PD spirals). We used popular image classification network weights, such as VGG-16, for transfer learning, and open source datasets such as the [UCI Machine Learning repository PD spiral dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet) and this Kaggle [dataset](https://www.kaggle.com/kmader/parkinsons-drawings). Transfer learning is the method of choice because the datasets of PD spirals are very small (on the order of ~50 images per class); we show that [INSERT QUANT RESULT HERE].

VIDEO GOES HERE (probably): [INSERT VIDEO HERE]

## Introduction

Parkinson's disease (PD) is a progressive neurological disorder with no cure and unknown etiology. Treatment can help slow disease progression and increase quality of life if symptoms are detected early enough. A common [clinical test](https://pn.bmj.com/content/17/6/456) involves the patient drawing on a piece of paper, which the physician then examines for evidence of early signs of PD tremors. This is a classification problem that can be aided by deep learning, particularly there is an opportunity here to introduce quantitative analysis on a currently qualitative task; if effective, such methods could perform large scale diagnostics (e.g. via a mobile phone app) to help detect PD in early stages and help researchers and clinicians to understand, track and ultimately treat PD more effectively than current methods allow.

The following sections summarize the results obtained by training a subset of popular CNN network layers (usually the last fully connected layer) and their performance of classification of PD spirals. We believe that it is important to find a way to easily detect PD symptoms from drawings in an accessible way; in the related work section, we describe efforts to classify PD tremors, however they use specialized tablets, etc. We believe the use of transfer learning would enable users to submit spirals directly through a touch screen or by uploading a hand drawn spiral for inference. Such an approach could increase physician reach and promote telemedicine/outpatient detection and tracking of PD symptoms among a larger portion of the population than currently achievable with the state-of-the-art.

## Related Work

Other's have approached the problem of PD spiral classification, however, they have done so with some key differences than the work presented here. [One study](https://ieeexplore.ieee.org/document/8064621), used a digital tablet to track various quantities of an individual's penmanship such as "angular features" (the angle of the pen to the tablet surface) and "direction inversion" (a measure of the change in angular features). The collection of features were then used to calculate a Spearman rank correlation coefficient, which supported the use of hand-drawn spirals for the detection of PD. This study did not provide a method of classification, but rather showed that features extracted from spirals are sufficient to distinguish people with and without PD.

[Another study](https://www.mdpi.com/2079-9292/8/8/907), directly investigated the use of CNNs for classification of PD spirals (I used their dataset!). The major difference of this work to the project presented here, is that they used frequency spectra (spectrograms) as input to the CNN rather than raw images. The researchers were able to create spectrograms because they collected data on a specialized tablet as well as the first study mentioned, rather than static images. The idea here is that tremors will become more apparent in the frequency domain, so the spectra are better inputs than raw images. Their results support this hypothesis as they achieved an accuracy of 96.5%, a F1-score of 97.7% and an area under the curve of 99.2%. This is the gold-standard performance to beat that I could find. The downside here is that patients would need access to a specialized tablet, which would likely require a normal doctor's office visit. This project aims to study the possibility of achieve such performance using only the static images of PD spirals rather than the spectra.

## Approach

We trained and compared 3 popular CNN-based architectures: ResNet-50, VGG-16 (with batch norm), and Inception_v3 (a.k.a. GoogleNetv3) on the task of PD spiral classification. We chose these architectures based their popularity and performance classifying on large image dataset such as ImageNet. The general approach was to use the pretrained network weights, and then train the final fully-connected layer from scratch on our training dataset. This is a common approach when using deep nets in practice/industry, we also explored training more than the final layer in some cases (more details on this later). We combined two Parkinson's spiral datasets: [UCI Machine Learning repository PD spiral dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet) and this Kaggle [dataset](https://www.kaggle.com/kmader/parkinsons-drawings). The Kaggle dataset was ready to use as-is, however, the UCI dataset needed significant cleaning and plotting to get into image format. We mixed the datasets and then randomly partitioned them into 70% training, 15% validation and 15% testing. We trained each model for 25 epochs, evaluating on the validation set after each epoch, and we saved the best performing model based on the validation set for final evaluation on the test set. We created graphs of training/validation loss and accuracy over the epochs. When evaluating the best performing models on the test set we computed accuracy, F1 score, precision and recall. Next, we describe each of these major components in greater detail.

### Model architectures

We used 3 popular architectures in the project: ResNet-50, VGG-16 and Inception_v3 (a.k.a GoogleNetv3). 

We first discuss the ResNet-50 architecture, here is a figure from the [ResNet paper](https://arxiv.org/abs/1512.03385) describing the architecture:

![image of the ResNet neural network architecture](https://github.com/minneker/transfer-learning-project/blob/main/images/resnet.png?raw=true)

The column to focus on is the "50-layer" column, which is the architecture used in this project. The architecture is essentially a 50 layer CNN, however, there are residual connections (i.e. shortcut-connections that skip one or more layers), in this network they perform an identity mapping. The main benefit to residual connections is they overcome the problem of degradation which is described by the authors succintly:

> "When deeper networks are able to start converging, a
degradation problem has been exposed: with the network
depth increasing, accuracy gets saturated (which might be
unsurprising) and then degrades rapidly. Unexpectedly,
such degradation is not caused by overfitting, and adding
more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by
our experiments. Fig. 1 shows a typical example."

We trained two variants of ResNet-50: 
- Froze all weights besides last FC
- Froze all weights besides last FC and the last convolutional block

We chose the first option because it is common practice in transfer learning to simply train the last fully connected layer. We chose the second option because early layers in CNNs tend to be more general feature extractors, and later layers tend to be more dataset specific; so our goal was to see if retraining the last convolutional block had a significant effect on the performance.

Next we have the VGG-16 architecture, here is a figure from the [VGG paper](https://arxiv.org/abs/1409.1556) describing the architecture:

![alt text](https://github.com/minneker/transfer-learning-project/blob/main/images/vgg.png?raw=true)

[INSERT DESCRIPTION HERE]

Lastly, we have the Inception_v3 architecture, here is a figure from the [Inception_v3 paper](https://arxiv.org/abs/1512.00567) describing the architecture:

![alt text](https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3.png?raw=true)

[INSERT DESCRIPTION HERE]

### Data collection and preparation

### Training setup 

### Evaluation metrics

How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?

## Results

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.
