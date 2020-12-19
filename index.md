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

The columns to focus on are the "C" and "D" configurations, which are the architectures used in this project. The architecture is essentially a 16 weight layer layer CNN, however, there are nuances that allow for strong network performance. For example, one such benefit is described by the authors for configuration C as the following:

> "The incorporation of 1 × 1 conv. layers (configuration C, Table 1) is a way to increase the nonlinearity of the decision function without affecting the receptive fields of the conv. layers. Even though in our case the 1 × 1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function. It should be noted that 1×1 conv. layers have recently been utilised in the “Network in Network” architecture of Lin et al. (2014)"

We trained two variants of VGG-16. The final output layer of VGG-16 has the following structure:

```
Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace=True)
  (5): Dropout(p=0.5, inplace=False)
  (6): Linear(in_features=4096, out_features=2, bias=True)
)
```

The variants we trained were:
- Froze all weights besides the very last FC layer, i.e. (6): Linear(in_features=4096, out_features=2, bias=True)
- Froze all weights besides the last two FC layers, i.e. (3) and (6)

We chose the first option because it is common practice in transfer learning to simply train the last fully connected layer. We chose the second option because we wanted to see if retraining the last two FC layers had a significant effect on the performance, similar to our experiments on ResNet-50.

Lastly, we have the Inception_v3 architecture, here is a figure from the [Inception_v3 paper](https://arxiv.org/abs/1512.00567) describing the architecture:

![alt text](https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3.png?raw=true)

The architecture is essentially a standard CNN, however, there are nuances that allow for strong network performance. For example, the Inception blocks are based on a principle of reducing large convolutions, which saves computation without sacrificing performance. Additionally, auxiliary classifiers are used to improve the convergence of deep networks such as Inception_v3. 

We trained two variants of Inception_v3:
- Froze all weights besides last fully connected layer
- Froze all weights besides last fully connected layer and auxiliary fully connected layer

We chose the first option because it is common practice in transfer learning to simply train the last fully connected layer. We chose the second option because we wanted to see if retraining the last auxiliary layer had a significant effect on the performance, similar to our experiments on ResNet-50 and VGG-16.

### Data collection and preparation

We used two Parkinson's spiral datasets as described earlier: [UCI Machine Learning repository PD spiral dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Spiral+Drawings+Using+Digitized+Graphics+Tablet) and this Kaggle [dataset](https://www.kaggle.com/kmader/parkinsons-drawings). The Kaggle dataset came fully prepared as images, however, the UCI dataset came as text files. We needed to extract the relevant information from the files and then plot the spirals to be used as input to the neural networks. You may see the data prepartion notebook in the main repo if you are interested in this process. Once combined, we randomly split the data into training (70%), validation (15%) and testing (15%). The following table summarizes the number of files in each category (Control and People with Parkinson's (PwP)):

- Training
  - Control: 48
  - PwP: 54
- Validation
  - Control: 9
  - PwP: 11
- Testing
  - Control: 9
  - PwP: 11
  
We used a `DataLoader` for each dataset (train/val/test), for ease of loading and transforming data to the correct input size, etc. For both VGG-16 and ResNet-50 we used the following dataloaders:

```
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4) for x in ['train', 'val']}
```

For Inception_v3, the network requires mini-batches of 3-channel RGB images of shape `(3 x H x W)`, `H` and `W` are expected to be at least 299. So our transformation changed to the following:

```
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

```

The test set loaders look identical to the validation loaders for all networks. Note we chose to have `batch_size=4` and `num_workers=4`. `batch_size` was kept constant for all experiments but future studies should definitely include variations of `batch_size` to measure their impact on transfer learning performance for this task.

### Training and evaluation setup

For training we followed the guidance of the transfer learning tutorial in the [PyTorch documentation](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Specifically, we used: a `Cross Entropy` loss function, `Stochastic Gradient Descent (SGD)` with `lr=0.001, momentum=0.9`, a learning rate scheduler to decay the learning rate by 0.1 every 7 epochs i.e. `gamma=0.1` and `step_size=7`. We trained each network for `25 epochs`, wherein we calculated `training/validation loss` and `training/validation accuracy` after each epoch. After 25 epochs, we saved the best model (based on validation set accuracy). This saved model was then loaded and used for inference on the test set where we calculated `accuracy`, `F1 score`, `precision`, `recall`.

The general model setup looked like:

```
# Load pretrained model and replace FC layer(s) to match our binary classification problem

# Require grad only on the layer(s) of interest

# Train the model!
```

For clarity we list all conditions tested below, where we assume the following are constant:

```
# Cross entropy loss function
criterion = nn.CrossEntropyLoss()

# SGD optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

```

#### ResNet-50

Last FC layer only:

```
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# Observe that __a subset of__ parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'fc.weight':
        params.requires_grad=False
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

model_name = 'resnet50_fc_weight_optim'
model_ft = train_model(model_ft, 
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler, 
                       model_name,
                       num_epochs=25)

torch.save(model_ft.state_dict(), model_dir + model_name)
```

Last FC layer and all of layer 4 (we only show the difference to avoid redundancy):

```
# Observe that **A subset of** parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'fc.weight' and not name.startswith('layer4'):
        params.requires_grad=False
```

#### Inception_v3

Last FC and Auxiliary FC:

```
model_ft = models.inception_v3(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

# Observe that **A subset of** parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'fc.weight' and name != 'AuxLogits.fc.weight':
        params.requires_grad=False

model_name = 'inceptionv3_fc_and_auxlogits_fc_optim'
model_ft = train_inception(model_ft, 
                           criterion, 
                           optimizer_ft, 
                           exp_lr_scheduler, 
                           model_name,
                           num_epochs=25)

torch.save(model_ft.state_dict(), model_dir + model_name)
```

Notice that we used `train_inception` here which accounted for the auxiliary loss as well during training, as follows:

```
# forward
# track history if only in train
with torch.set_grad_enabled(phase == 'train'):
    if phase == 'val':
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, preds)
    else:  # in training mode
        outputs, aux_outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss1 = criterion(outputs, preds)
        loss2 = criterion(aux_outputs, preds)
        loss = loss1 + 0.4*loss2

        # backward + optimize only if in training phase
        loss.backward() 
        optimizer.step()
```

The combined loss function is necessay when training the auxiliary layers, that is why the combined loss function was used in this case. Note that `aux_logits=True` by default in the model declaration so they are used in this `Inception_v3` instance.

Last FC only:

```
model_ft = models.inception_v3(pretrained=True, aux_logits=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)


# Observe that **A subset of** parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'fc.weight':
        params.requires_grad=False

model_name = 'inceptionv3_fc_optim'
model_ft = train_model(model_ft, 
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler, 
                       model_name,
                       num_epochs=25)

torch.save(model_ft.state_dict(), model_dir + model_name)
```
Note that `aux_logits=False` is used so we no longer need a combined loss function since the auxiliary layers are no longer used in training so we call `train_model` instead of `train_inception`.

#### VGG-16

Last FC only:

```
model_ft = models.vgg16_bn(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, 2) # replace last fc layer
model_ft = model_ft.to(device)


# Observe that **A subset of** parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'classifier.6.weight':
        params.requires_grad=False

model_name = 'vgg16_bn_final_fc_weight'
model_ft = train_model(model_ft, 
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler, 
                       model_name,
                       num_epochs=25)

torch.save(model_ft.state_dict(), model_dir + model_name)
```

Last two FC layers:

```
model_ft = models.vgg16_bn(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[3] = nn.Linear(num_ftrs, num_ftrs) # replace 2nd to last layer
model_ft.classifier[6] = nn.Linear(num_ftrs, 2) # replace last fc layer
model_ft = model_ft.to(device)


# Observe that **A subset of** parameters are being optimized
for name, params in model_ft.named_parameters():
    if name != 'classifier.6.weight' and name != 'classifier.3.weight':
        params.requires_grad=False

model_name = 'vgg16_bn_last_two_fc_weights'
model_ft = train_model(model_ft, 
                       criterion, 
                       optimizer_ft, 
                       exp_lr_scheduler, 
                       model_name,
                       num_epochs=25)

torch.save(model_ft.state_dict(), model_dir + model_name)
```


### Evaluation metrics

For each of the models in the previous section we plotted the training and validation loss and accuracy curves over the 25 epochs. The best performing model based on validation accuracy from the 25 epochs was evaluated against the test set via `F1 score`, `accuracy`, `precision` and `recall`. The results are summarized in the next section.

## Results

Here are the `F1 score`, `accuracy`, `precision` and `recall` for each model evaluated against the test set:
  
| Model                             | F1            | Accuracy  | Precision   | Recall   |
| ----------------------------------|---------------|-----------|-------------|----------|
| ResNet-50 (last FC)               | 0.7778        | **0.8000**|**1.0000**   |0.6364    |
| ResNet-50 (last FC & layer 4)     | 0.7619        | 0.7500    |0.8000       |0.7273    |
| Inception_v3 (FC)                 | **0.8333**    | **0.8000**|0.7692       |0.9091    |
| Inception_v3 (FC and aux)         | 0.7097        | 0.5500    |0.5500       |**1.0000**|
| VGG-16 (last FC)                  | 0.6316        | 0.6500    |0.7500       |0.5455    |
| VGG-16 (last two FCs)             | 0.7407        | 0.6500    |0.6250       |0.9091    |

Here are the plots of accuracy and loss over the training epochs for each of the model architectures:

### ResNet-50 (last FC)

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/resnet50_fc_weight_optim_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/resnet50_fc_weight_optim_loss.png?raw=true" width="49%" /> 
</p>

### ResNet-50 (last FC & layer 4) 

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/resnet50_fc_weight_and_layer4_optim_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/resnet50_fc_weight_and_layer4_optim_loss.png?raw=true" width="49%" /> 
</p>

### Inception_v3 (FC)

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3_fc_optim_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3_fc_optim_loss.png?raw=true" width="49%" /> 
</p>

### Inception_v3 (FC and aux)

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3_fc_and_auxlogits_fc_optim_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/inceptionv3_fc_and_auxlogits_fc_optim_loss.png?raw=true" width="49%" /> 
</p>

### VGG-16 (last FC)

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/vgg16_bn_final_fc_weight_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/vgg16_bn_final_fc_weight_loss.png?raw=true" width="49%" /> 
</p>

### VGG-16 (last two FCs) 

<p float="left">
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/vgg16_bn_last_two_fc_weights_acc.png?raw=true" width="49%" />
  <img src="https://github.com/minneker/transfer-learning-project/blob/main/images/vgg16_bn_last_two_fc_weights_loss.png?raw=true" width="49%" /> 
</p>


## Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.
