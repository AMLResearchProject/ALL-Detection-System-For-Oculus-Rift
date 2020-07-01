# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

### AllDS2020 CNN

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [ALL-IDB](#all-idb)
  - [ALL_IDB1](#all_idb1)
- [Network Architecture](#network-architecture)
  - [Results Overview](#results-overview)
- [Installation](#installation)
    - [Anaconda](#anaconda)
    - [Clone The Repository](#clone-the-repository)
    - [Setup File](#setup-file)
    - [Windows Installation Issues](#windows-installation-issues)
    - [AutoPilot Scripts](#autopilot-scripts)
- [Getting Started](#getting-started)
    - [Data](#data)
    - [Code Structure](#code-structure)
        - [Classes](#classes)
            - [Functions](#functions)
    - [Configuration](#configuration)
- [Metrics](#metrics)
- [Training The Model](#training-the-model)
    - [Start The Training](#start-the-training)
        - [Data](#data)
        - [Model](#model)
        - [Training Results](#training-results)
        - [Metrics Overview](#metrics-overview)
        - [ALL-IDB Required Metrics](#all-idb-required-metrics)
- [Local Testing](#local-testing)
    - [Local Testing Results](#local-testing-results)
- [Server Testing](#server-testing)
    - [Server Testing Results](#server-testing-results)
- [Raspberry Pi 4](#raspberry-pi-4)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction

This project trains the model that will be used in Acute the Lymphoblastic Leukemia Detection System 2020. The network provided in this project was originally created in [ALL research papers evaluation project](https://github.com/LeukemiaAiResearch/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Evaluations/Paper-1.md "ALL research papers evaluation project"), where we replicated the network proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the data augmentation proposed in  [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The original project was inspired by the [work](https://github.com/AmlResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb "work") done by [Amita Kapoor](https://www.petermossamlallresearch.com/team/amita-kapoor/profile "Amita Kapoor") and [Taru Jain](https://www.petermossamlallresearch.com/students/student/taru-jain/profile "Taru Jain") and my [projects](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Keras/AllCNN "projects") based on their work.

&nbsp;

# DISCLAIMER

This project should be used for research purposes only. The purpose of the project is to show the potential of Artificial Intelligence for medical support systems such as diagnosis systems. Although the program is fairly accurate and shows good results both on paper and in real world testing, it is not meant to be an alternative to professional medical diagnosis. I am a self taught developer with some experience in using Artificial Intelligence for detecting certain types of cancer. I am not a doctor, medical or cancer expert. Please use this system responsibly.

&nbsp;

# ALL-IDB

You need to be granted access to use the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. You can find the application form and information about getting access to the dataset on [this page](https://homes.di.unimi.it/scotti/all/#download) as well as information on how to contribute back to the project [here](https://homes.di.unimi.it/scotti/all/results.php). If you are not able to obtain a copy of the dataset please feel free to try this tutorial on your own dataset, we would be very happy to find additional AML & ALL datasets.

## ALL_IDB1 

In this project, [ALL-IDB1](https://homes.di.unimi.it/scotti/all/#datasets) is used, one of the datsets from the Acute Lymphoblastic Leukemia Image Database for Image Processing dataset. We will use data augmentation to increase the amount of training and testing data we have.

"The ALL_IDB1 version 1.0 can be used both for testing segmentation capability of algorithms, as well as the classification systems and image preprocessing methods. This dataset is composed of 108 images collected during September, 2005. It contains about 39000 blood elements, where the lymphocytes has been labeled by expert oncologists. The images are taken with different magnifications of the microscope ranging from 300 to 500."

&nbsp;

# Network Architecture

<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"), the authors propose a simple 5 layer Convolutional Neural Network. 

In this project we will use an augmented dataset with the network proposed in this paper, built using Tensorflow 2.

We will build a Convolutional Neural Network, as shown in Fig 1, consisting of the following 5 layers (missing out the zero padding layers). Note we are usng an conv sizes of (100x100x30) whereas in the paper, the authors use (50x50x30).

- Conv layer (100x100x30)
- Conv layer (100x100x30)
- Max-Pooling layer (50x50x30)
- Fully Connected layer (2 neurons)
- Softmax layer (Output 2)

## Results Overview

We have tested our model on a number of different hardwares, including Intel® CPUs & NVIDIA GPUs. Results seem to vary between CPU & GPU, and tests show further investigation into seeding and randomness introduced to our network via the GPU software. For reproducible results every time, it is suggested to train on a CPU, although this obviously requires more time.

One method to overcome reproducibility issues and get an good idea of how well our model is behaving on GPU would be to test the model multiple times and take an average. This is one way we will explore our model in the future.

Below are the results from individual training sessions.


| OS | Hardware | Training | Validation | Test | Accuracy | Recall | Precision | AUC/ROC |
| -------------------- | -------------------- | -------------------- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Google Colab | Tesla K80 GPU | 1180 |  404 | 20 |  0.9727723 | 0.9727723 | 0.9727723 | 0.9948964 |
| Windows 10 | NVIDIA GeoForce GTX 1060 | 1180 |  404 | 20 |  0.97066015 | 0.97066015 | 0.97066015 | 0.9908836 |
| Ubuntu 18.04 | NVIDIA GTX 1050 Ti Ti/PCIe/SSE2 | 1180 |  404 | 20 |  0.97772276 | 0.97772276 | 0.97772276 | 0.9989155 |
| Ubuntu 18.04 | Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8   | 1180 |  404 | 20 |  0.9752475 | 0.9752475 | 0.9752475 | 0.991492 |
| Windows 10 | Intel® Core™ i7-7700HQ CPU @ 2.80GHz × 8   | 1180 |  404 | 20 |  0.9851485 | 0.9851485 | 0.9851485 | 0.9985846 |
| macOS Mojave 10.14.6 | Intel® Core™ i5 CPU @ 2.4 GHz   | 1180 |  404 | 20 |  0.9589041 | 0.9589041 | 0.9589041 | 0.99483955 |

&nbsp;

# Installation

## Anaconda

If you haven't already installed Anaconda and set up your conda env and Tensorflow installation, please follow our [Anaconda installation guide](../Documentation/Anaconda.md "Anaconda installation guide"). 

## Clone the repository

Clone the [Acute Lymphoblastic Leukemia Detection System 2020](https://github.com/AMLResearchProject/ALL-Detection-System-2020 " Acute Lymphoblastic Leukemia Detection System 2020") repository from the [Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project](https://github.com/AMLResearchProject "Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project") Github Organization.

To clone the repository and install the Acute Lymphoblastic Leukemia Detection System 2020 Classifier For Raspberry Pi 4, make sure you have Git installed. Now navigate to the home directory on your device using terminal/commandline, and then use the following command.

```
$ git clone https://github.com/AMLResearchProject/ALL-Detection-System-2020.git
```

Once you have used the command above you will see a directory called **ALL-Detection-System-2020** in your home directory.

```
ls
```

Using the ls command in your home directory should show you the following.

```
ALL-Detection-System-2020
```

Navigate to **ALL-Detection-System-2020/RPI4** directory, this is your project root directory for this tutorial.

### Developer Forks

Developers from the Github community that would like to contribute to the development of this project should first create a fork, and clone that repository. For detailed information please view the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") guide. You should pull the latest code from the development branch.

```
git clone -b "0.3.0" https://github.com/COVID-19-AI-Research-Project/AI-Classification.git
```

The **-b "0.3.0"** parameter ensures you get the code from the latest master branch. Before using the below command please check our latest master branch in the button at the top of the project README.

## Setup File

All other requirements are included in **Setup.sh**. You can run this file on machine by navigating to the **CNN** directory in terminal and using the command below:

```
sh Setup.sh
```

## Windows Installation Issues

If you're working on a Windows 10 machine and facing some issues, please follow our [Windows Issues guide](../Documentation/Windows.md "Windows Issues guide"). In case your issue is not mentioned and you're able to solve it, do create a pull request mentioning the same in the aforementioned file.

## AutoPilot Scripts

If you would like to replicate the exact scenarios we tested in or simply like to put the entire process in AutoPilot, please follow our [AutoPilot guide](../Documentation/AutoPilot.md "AutoPilot guide"). 

&nbsp;

# Getting Started

## Data

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **Model/Data**, inside you have **Train** & **Test**. 

We will created an augmented dataset based on the [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. In this case, we will use more rotated images to increase the dataset further.

First take the ten positive and ten negative samples shown below, and place them in the **Model/Data/Test** directory. This will be used by our Oculus Rift application and our testing purposes. You can use any data split you like, to ensure you get the same results please use the same test images and the CPU for training. There is currently an issue when using GPU, most likely due to randomness implemented in the GPU software. It is possible to replicate the results here but it may take more than one attempt at training. This is something we will work on for a future update.

- im006_1.jpg
- im020_1.jpg
- im024_1.jpg
- im026_1.jpg
- im028_1.jpg
- im031_1.jpg
- im035_0.jpg
- im041_0.jpg
- im047_0.jpg
- im053_1.jpg
- im057_1.jpg
- im060_1.jpg
- im063_1.jpg
- im069_0.jpg
- im074_0.jpg
- im088_0.jpg
- im095_0.jpg
- im099_0.jpg
- im101_0.jpg
- im106_0.jpg

Next add the remaining 88 images to the **Model/Data/Train** folder. The test images used will not be augmented.

## Code structure

The code for this project consists of 5 main Python files and a configuration file:

- [config.json](Model/config.json "config.json"): The configuration file.
- [AllDS2020.py](AllDS2020.py "AllDS2020.py"): Core classifier wrapper class.
- [Helpers.py](Classes/Helpers.py "Helpers.py"): A helpers class.
- [Data.py](Classes/Data.py "Data.py"): A data helpers class.
- [Augmentation.py](Classes/Augmentation.py "Augmentation.py"): An augmentation helpers class.
- [Model.py](Classes/Model.py "Model.py"): A model helpers class.
- [Server.py](Classes/Server.py "Server.py"): A server helpers class.

### Classes 

Our functionality for this network can be found mainly in the **Classes** directory. 

- [Helpers.py](Classes/Helpers.py "Helpers.py") is a helper class. The class loads the configuration and logging that the project uses.
- [Data.py](Classes/Data.py "Data.py") is a data helper class. The class provides the functionality for sorting and preparing your training and validation data.
- [Augmentation.py](Classes/Augmentation.py "Augmentation.py") is a augmentation helper class, The class provides functionality for data augmentation.
- [Model.py](Classes/Model.py "Model.py") is a model helper class. The class provides the functionality for creating our CNN. 
- [Server.py](Classes/Server.py "Server.py") is a server helpers class. The class provides the functionality for creating our CNN 

#### Functions

 The main functions are briefly explained below:

 ##### Data.py

- **do_im_process()** - The do_im_process() function augments and prepares the data.
- **convert_data()** - The convert_data() function converts the training data to a numpy array.
- **encode_labels()** - The encode_labels() function One Hot Encodes the labels.
- **shuffle()** - The shuffle() function shuffles the data helping to eliminate bias.
- **get_split()** - The get_split() function splits the prepared data and labels into training and validation data.
- **resize()** - The resize() function resizes an image.

 ##### Augmentation.py

- **grayscale()** The grayscale() function creates a grayscale copy of an image.
- **equalize_hist()** The equalize_hist() function creates a histogram equalized copy of an image.
- **reflection()** The reflection() function creates a horizontally and vertically reflected copies of an image.
- **gaussian()** The gaussian() function creates a gaussian blurred copy of an image.
- **translate()** The translate() function creates a translated copy of an image.
- **rotation()** The rotation() function creates rotated copy/copies of an image.
- **shear()** The shear() function creates sheared copy of an image.

 ##### Model.py

- **do_data()** The do_data() creates an augmented dataset that we will use for our model training and validation.
- **do_network()** The do_network() function creates the network architecture proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper.
- **do_train()** The do_train() function compiles and trains the model.
- **do_evaluate()** The do_evaluate() function evaluates the model, and displays the values for the metrics we specified.
- **do_predictions()** The do_predictions() function makes predictions on the train & test sets.
- **visualize_metrics()** The visualize_metrics() function allows you to visualize and save the metrics plot images.
- **confusion_matrix()** The confusion_matrix() function allows you to visualize the confusion matrix.
- **figures_of_merit()** The figures_of_merit() calculates the figures of merit requested on the ALL-IDB website.
- **save_weights()** The save_weights() saves the model weights.
- **save_model_as_json()** The save_model_as_json() saves the model as JSON.
- **load_model_and_weights()** The load_model_and_weights() loads the trained model and weights.
- **test_classifier()** The test_classifier() tests the classifier using the test data set.
- **send_request()** The send_request() sends a HTTP request.
- **test_http_classifier()** The test_http_classifier() tests the server / API by sending the test data to the classifier with the API.
- **http_classify()** The http_classify() classifies an image sent via HTTP.
- **vr_http_classify()** The vr_http_classify() classifies an image sent via from VR via HTTP.
- **get_predictions()** The get_predictions() gets a prediction for an image.
- **reshape()** The reshape() reshapes an image.

 ##### Server.py

- **start()** The start() starts the classification API server.

&nbsp;

## Configuration

[config.json](Model/config.json "config.json")  holds the configuration for our network. 

```
{
    "cnn": {
        "api": {
            "server": "XXX.XXX.X.XXX",
            "port": 1234
        },
        "core": [
            "Train",
            "Server",
            "Client",
            "Classify"
        ],
        "data": {
            "dim": 100,
            "file_type": ".jpg",
            "labels": [0,1],
            "rotations": 10,
            "seed": 2,
            "split": 0.3,
            "test": "Model/Data/Test",
            "test_data": [
                "im006_1.jpg",
                "im020_1.jpg",
                "im024_1.jpg",
                "im026_1.jpg",
                "im028_1.jpg",
                "im031_1.jpg",
                "im035_0.jpg",
                "im041_0.jpg",
                "im047_0.jpg",
                "im053_1.jpg",
                "im057_1.jpg",
                "im060_1.jpg",
                "im063_1.jpg",
                "im069_0.jpg",
                "im074_0.jpg",
                "im088_0.jpg",
                "im095_0.jpg",
                "im099_0.jpg",
                "im101_0.jpg",
                "im106_0.jpg"
            ],
            "train_dir": "Model/Data/Train",
            "valid_types": [
              ".JPG",
              ".JPEG",
              ".PNG",
              ".GIF",
              ".jpg",
              ".jpeg",
              ".png",
              ".gif"
            ]
        },
        "model": {
            "model": "Model/model.json",
            "weights": "Model/weights.h5"
        },
        "train": {
            "batch": 100,
            "decay_adam": 1e-6,
            "epochs": 150,
            "learning_rate_adam": 1e-4,
            "val_steps": 10
        }
    }
}
```

The cnn object contains 4 Json Objects (api, data, model and train) and a JSON Array (core). Api has the information used to set up your server you will need to add your local ip, data has the configuration related to preparing the training and validation data, model holds the model file paths, and train holds the training parameters. 

In my case, the configuration above was the best out of my testing, but you may find different configurations work better. Feel free to update these settings to your liking, and please let us know of your experiences.

&nbsp;

# Metrics

We can use metrics to measure the effectiveness of our model. In this network we will use the following metrics:

```
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc')
```

These metrics will be displayed and plotted once our model is trained.  A useful tutorial while working on the metrics was the [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial on Tensorflow's website.

&nbsp;

# Training the model

Now you are ready to train your model. As mentioned above, an Ubuntu machine with an NVIDIA GTX 1050 ti was used. Using different machines/GPU(standalone or integrated)/CPU may vary the results, if so please let us know your findings.

There is a known issue when training on NVIDIA GPU/CUDA that introduces randomness to the training process and ultimately, the results. For reproducible results every time, it is suggested to train on a CPU, although this obviously requires more time.

One method to overcome reproducibility issues and get an good idea of how well our model is behaving on GPU would be to test the model multiple times and take an average. This is one way we will explore our model in the future.

## Start The Training

Ensuring you have completed all previous steps, you can start training using the following command. 

```
python AllDS2020.py Train
```

This tells the classifier to start in Train mode which will start the model training process.

### Data

First the data will be prepared.

```
2020-07-01 19:34:28,273 - Data - INFO - Data shape: (1584, 100, 100, 3)
2020-07-01 19:34:28,274 - Data - INFO - Labels shape: (1584, 2)
2020-07-01 19:34:28,274 - Data - INFO - Raw data: 792
2020-07-01 19:34:28,274 - Data - INFO - Raw negative data: 441
2020-07-01 19:34:28,275 - Data - INFO - Raw positive data: 792
2020-07-01 19:34:28,275 - Data - INFO - Augmented data: (1584, 100, 100, 3)
2020-07-01 19:34:28,275 - Data - INFO - Labels: (1584, 2)
2020-07-01 19:34:28,343 - Data - INFO - Training data: (1180, 100, 100, 3)
2020-07-01 19:34:28,343 - Data - INFO - Training labels: (1180, 2)
2020-07-01 19:34:28,343 - Data - INFO - Validation data: (404, 100, 100, 3)
2020-07-01 19:34:28,343 - Data - INFO - Validation labels: (404, 2)
2020-07-01 19:34:28,344 - Model - INFO - Data preperation complete.
```

### Model Summary

Our network matches the architecture proposed in the paper exactly, with exception to maybe the optimizer and loss function as this info was not provided in the paper. 

Before the model begins training, we will be shown the model summary, or architecture. 

```
Model: "AllDS2020_TF_CNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
zero_padding2d (ZeroPadding2 (None, 104, 104, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 30)      2280
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 104, 104, 30)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 30)      22530
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 30)        0
_________________________________________________________________
flatten (Flatten)            (None, 75000)             0
_________________________________________________________________
dense (Dense)                (None, 2)                 150002
_________________________________________________________________
activation (Activation)      (None, 2)                 0
=================================================================
Total params: 174,812
Trainable params: 174,812
Non-trainable params: 0
_________________________________________________________________

Train on 1180 samples, validate on 404 samples
```

## Training Results

Below are the training results for 150 epochs.

<img src="Model/Plots/Accuracy.png" alt="Adam Optimizer Results" />

_Fig 2. Ubuntu/GTX 1050 ti Accuracy_

<img src="Model/Plots/Loss.png" alt="Ubuntu/GTX 1050 ti Loss" />

_Fig 3. Ubuntu/GTX 1050 ti Loss_

<img src="Model/Plots/Precision.png" alt="Ubuntu/GTX 1050 ti Precision" />

_Fig 4. Ubuntu/GTX 1050 ti Precision_

<img src="Model/Plots/Recall.png" alt="Ubuntu/GTX 1050 ti Recall" />

_Fig 5. Ubuntu/GTX 1050 ti Recall_

<img src="Model/Plots/AUC.png" alt="Ubuntu/GTX 1050 ti AUC" />

_Fig 6. Ubuntu/GTX 1050 ti AUC_

```2020-07-01 19:39:05,684 - Model - INFO - Metrics: loss 0.04732655737512183
2020-07-01 19:39:05,684 - Model - INFO - Metrics: acc 0.97772276
2020-07-01 19:39:05,684 - Model - INFO - Metrics: precision 0.97772276
2020-07-01 19:39:05,684 - Model - INFO - Metrics: recall 0.97772276
2020-07-01 19:39:05,684 - Model - INFO - Metrics: auc 0.9989155

2020-07-01 19:39:06,117 - Model - INFO - Confusion Matrix: [[230   4]
 [  5 165]]

2020-07-01 19:39:06,206 - Model - INFO - True Positives: 165(40.84158415841584%)
2020-07-01 19:39:06,206 - Model - INFO - False Positives: 4(0.9900990099009901%)
2020-07-01 19:39:06,206 - Model - INFO - True Negatives: 230(56.93069306930693%)
2020-07-01 19:39:06,206 - Model - INFO - False Negatives: 5(1.2376237623762376%)
2020-07-01 19:39:06,206 - Model - INFO - Specificity: 0.9829059829059829
2020-07-01 19:39:06,206 - Model - INFO - Misclassification: 9(2.227722772277228%)
```

## Metrics Overview

| Accuracy | Recall | Precision | AUC/ROC |
| ---------- | ---------- | ---------- | ---------- |
| 0.97772276 | 0.97772276 | 0.97772276 | 0.9989155 |

## ALL-IDB Required Metrics

| Figures of merit     | Amount/Value | Percentage |
| -------------------- | ----- | ---------- |
| True Positives       | 165 | 40.84158415841584% |
| False Positives      | 4 | 0.9900990099009901% |
| True Negatives       | 230 | 56.93069306930693% |
| False Negatives      | 5 | 1.2376237623762376% |
| Misclassification    | 9 | 2.227722772277228% |
| Sensitivity / Recall | 0.97772276   | 0.98% |
| Specificity          | 0.9829059829059829  | 99% |

&nbsp;

# Local Testing

Now we will use the test data to see how the classifier reacts to our testing data. Real world testing is the most important testing, as it allows you to see the how the model performs in a real world environment. 

This part of the system will use the test data from the **Model/Data/Test** directory. The command to start testing locally is as follows:

```
python AllDS2020.py Classify
```

## Local Testing Results

```
2020-07-01 19:45:45,650 - Model - INFO - Loaded test image Model/Data/Test/Im060_1.jpg
2020-07-01 19:45:45.700323: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-07-01 19:45:45.832584: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-01 19:45:46,439 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:46,519 - Model - INFO - Loaded test image Model/Data/Test/Im099_0.jpg
2020-07-01 19:45:46,538 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:46,571 - Model - INFO - Loaded test image Model/Data/Test/Im006_1.jpg
2020-07-01 19:45:46,589 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:46,665 - Model - INFO - Loaded test image Model/Data/Test/Im101_0.jpg
2020-07-01 19:45:46,682 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:46,714 - Model - INFO - Loaded test image Model/Data/Test/Im024_1.jpg
2020-07-01 19:45:46,733 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:46,806 - Model - INFO - Loaded test image Model/Data/Test/Im074_0.jpg
2020-07-01 19:45:46,823 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-07-01 19:45:46,897 - Model - INFO - Loaded test image Model/Data/Test/Im041_0.jpg
2020-07-01 19:45:46,914 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:46,988 - Model - INFO - Loaded test image Model/Data/Test/Im047_0.jpg
2020-07-01 19:45:47,005 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-07-01 19:45:47,036 - Model - INFO - Loaded test image Model/Data/Test/Im020_1.jpg
2020-07-01 19:45:47,053 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,127 - Model - INFO - Loaded test image Model/Data/Test/Im035_0.jpg
2020-07-01 19:45:47,144 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:47,218 - Model - INFO - Loaded test image Model/Data/Test/Im069_0.jpg
2020-07-01 19:45:47,235 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:47,267 - Model - INFO - Loaded test image Model/Data/Test/Im026_1.jpg
2020-07-01 19:45:47,284 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,358 - Model - INFO - Loaded test image Model/Data/Test/Im057_1.jpg
2020-07-01 19:45:47,375 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,449 - Model - INFO - Loaded test image Model/Data/Test/Im053_1.jpg
2020-07-01 19:45:47,469 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)
2020-07-01 19:45:47,501 - Model - INFO - Loaded test image Model/Data/Test/Im028_1.jpg
2020-07-01 19:45:47,517 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,584 - Model - INFO - Loaded test image Model/Data/Test/Im095_0.jpg
2020-07-01 19:45:47,601 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-07-01 19:45:47,675 - Model - INFO - Loaded test image Model/Data/Test/Im088_0.jpg
2020-07-01 19:45:47,692 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)
2020-07-01 19:45:47,766 - Model - INFO - Loaded test image Model/Data/Test/Im063_1.jpg
2020-07-01 19:45:47,783 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,815 - Model - INFO - Loaded test image Model/Data/Test/Im031_1.jpg
2020-07-01 19:45:47,832 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)
2020-07-01 19:45:47,905 - Model - INFO - Loaded test image Model/Data/Test/Im106_0.jpg
2020-07-01 19:45:47,922 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)
2020-07-01 19:45:47,923 - Model - INFO - Images Classifier: 20
2020-07-01 19:45:47,923 - Model - INFO - True Positives: 9
2020-07-01 19:45:47,923 - Model - INFO - False Positives: 4
2020-07-01 19:45:47,923 - Model - INFO - True Negatives: 6
2020-07-01 19:45:47,923 - Model - INFO - False Negatives: 1
```

&nbsp;

# Server Testing

Now we will use the test data to see how the server classifier reacts.

This part of the system will use the test data from the **Model/Data/Test** directory. 

You need to open two terminal windows or tabs, in the first, use the following command to start the server:

```
python AllDS2020.py Server
```

In your second terminal, use the following command:

```
python AllDS2020.py Client
```

## Server Testing Results

```
python AllDS2020.py Client
2020-07-01 19:51:53,161 - Core - INFO - Helpers class initialization complete.
2020-07-01 19:51:53,163 - Model - INFO - Model class initialization complete.
2020-07-01 19:51:53,163 - Core - INFO - AllDS2020 CNN initialization complete.
2020-07-01 19:51:53,164 - Model - INFO - Sending request for: Model/Data/Test/Im028_1.jpg
2020-07-01 19:51:54,108 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:52:01,114 - Model - INFO - Sending request for: Model/Data/Test/Im060_1.jpg
2020-07-01 19:52:02,645 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:52:09,653 - Model - INFO - Sending request for: Model/Data/Test/Im057_1.jpg
2020-07-01 19:52:11,162 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:52:18,169 - Model - INFO - Sending request for: Model/Data/Test/Im041_0.jpg
2020-07-01 19:52:19,672 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:52:26,680 - Model - INFO - Sending request for: Model/Data/Test/Im106_0.jpg
2020-07-01 19:52:28,191 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:52:35,199 - Model - INFO - Sending request for: Model/Data/Test/Im101_0.jpg
2020-07-01 19:52:36,715 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:52:43,723 - Model - INFO - Sending request for: Model/Data/Test/Im088_0.jpg
2020-07-01 19:52:45,229 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)

2020-07-01 19:52:52,236 - Model - INFO - Sending request for: Model/Data/Test/Im026_1.jpg
2020-07-01 19:52:53,060 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:53:00,068 - Model - INFO - Sending request for: Model/Data/Test/Im031_1.jpg
2020-07-01 19:53:00,880 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:53:07,888 - Model - INFO - Sending request for: Model/Data/Test/Im024_1.jpg
2020-07-01 19:53:08,705 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:53:15,713 - Model - INFO - Sending request for: Model/Data/Test/Im099_0.jpg
2020-07-01 19:53:17,206 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:53:24,213 - Model - INFO - Sending request for: Model/Data/Test/Im020_1.jpg
2020-07-01 19:53:25,032 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:53:32,039 - Model - INFO - Sending request for: Model/Data/Test/Im047_0.jpg
2020-07-01 19:53:33,543 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:53:40,551 - Model - INFO - Sending request for: Model/Data/Test/Im053_1.jpg
2020-07-01 19:53:42,070 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly not detected (False Negative)

2020-07-01 19:53:49,077 - Model - INFO - Sending request for: Model/Data/Test/Im006_1.jpg
2020-07-01 19:53:49,893 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:53:56,901 - Model - INFO - Sending request for: Model/Data/Test/Im074_0.jpg
2020-07-01 19:53:58,386 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)

2020-07-01 19:54:05,393 - Model - INFO - Sending request for: Model/Data/Test/Im069_0.jpg
2020-07-01 19:54:06,901 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:54:13,909 - Model - INFO - Sending request for: Model/Data/Test/Im063_1.jpg
2020-07-01 19:54:15,425 - Model - INFO - Acute Lymphoblastic Leukemia correctly detected (True Positive)

2020-07-01 19:54:22,433 - Model - INFO - Sending request for: Model/Data/Test/Im035_0.jpg
2020-07-01 19:54:23,941 - Model - INFO - Acute Lymphoblastic Leukemia correctly not detected (True Negative)

2020-07-01 19:54:30,949 - Model - INFO - Sending request for: Model/Data/Test/Im095_0.jpg
2020-07-01 19:54:32,366 - Model - INFO - Acute Lymphoblastic Leukemia incorrectly detected (False Positive)

2020-07-01 19:54:39,374 - Model - INFO - Images Classifier: 20
2020-07-01 19:54:39,375 - Model - INFO - True Positives: 9
2020-07-01 19:54:39,375 - Model - INFO - False Positives: 3
2020-07-01 19:54:39,375 - Model - INFO - True Negatives: 7
2020-07-01 19:54:39,376 - Model - INFO - False Negatives: 1
```

&nbsp;

# Raspberry Pi 4

Now that your model is trained and tested, head over to the [RPI 4](../RPI4 "RPI 4") project to setup your model on the Raspberry Pi 4 ready to be used with [HIAS](https://github.com/LeukemiaAiResearch/HIAS "HIAS") and the [Rift](../Rift "Rift") Virtual Reality classifier.

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

- [Javier Lopez Alonso](https://www.leukemiaresearchassociation.ai/team/javier-lopez-alonso "Javier Lopez Alonso") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Treasurer/Research & Development, Sabadell, Spain

- [Rishabh Banga](https://www.leukemiaresearchassociation.ai/team/rishabh-banga "Rishabh Banga") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Research & Development, Delhi, India

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE.md "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.