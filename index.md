## Welcome to my final project page!

We set out to use deep transfer learning with popular convolutional neural network architectures to diagnose Parkinson's disease (PD) from images of handwritten spirals (i.e. classify healthy vs. PD spirals). We used popular image classification network weights, such as VGG-16, for transfer learning, and open source datasets such as the [UCI Machine Learning repository PD spiral dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet) and this Kaggle [dataset](https://www.kaggle.com/kmader/parkinsons-drawings). Transfer learning is the method of choice because the datasets of PD spirals are very small (on the order of ~50 images per class); we show that [INSERT QUANT RESULT HERE].

VIDEO GOES HERE (probably): [INSERT VIDEO HERE]

## Introduction

Parkinson's disease (PD) is a progressive neurological disorder with no cure and unknown etiology. Treatment can help slow disease progression and increase quality of life if symptoms are detected early enough. A common [clinical test](https://pn.bmj.com/content/17/6/456) involves the patient drawing on a piece of paper, which the physician then examines for evidence of early signs of PD tremors. This is a classification problem that can be aided by deep learning, particularly there is an opportunity here to introduce quantitative analysis on a currently qualitative task; if effective, such methods could perform large scale diagnostics (e.g. via a mobile phone app) to help detect PD in early stages and help researchers and clinicians to understand, track and ultimately treat PD more effectively than current methods allow.

The following sections summarize the results obtained by training a subset of popular CNN network layers (usually the last fully connected layer) and their performance of classification of PD spirals. We believe that it is important to find a way to easily detect PD symptoms from drawings in an accessible way; in the related work section, we describe efforts to classify PD tremors, however they use specialized tablets, etc. We believe the use of transfer learning would enable users to submit spirals directly through a touch screen or by uploading a hand drawn spiral for inference. Such an approach could increase physician reach and promote telemedicine/outpatient detection and tracking of PD symptoms among a larger portion of the population than currently achievable with the state-of-the-art.

## Related Work

Other people are out there doing things. What did they do? Was it good? Was it bad? Talk about it here.

## Approach

How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?

## Results

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.
