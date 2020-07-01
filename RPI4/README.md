# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

### Tensorflow 2.0 AllDS2020 CNN For Raspberry Pi 4

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [ALL-IDB](#all-idb)
  - [ALL_IDB1](#all_idb1)
- [Network Architecture](#network-architecture)
  - [Results Overview](#results-overview)
- [Installation](#installation)
    - [CNN](#cnn)
    - [Data](#data)
    - [Setup File](#setup-file)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction

This project is the classifier that is used in Acute the Lymphoblastic Leukemia Detection System 2020. The network provided in this project was originally created in [ALL research papers evaluation project](https://github.com/LeukemiaAiResearch/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Evaluations/Paper-1.md "ALL research papers evaluation project"), where we replicated the network proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the data augmentation proposed in  [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The original project was inspired by the [work](https://github.com/AmlResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb "work") done by [Amita Kapoor](https://www.petermossamlallresearch.com/team/amita-kapoor/profile "Amita Kapoor") and [Taru Jain](https://www.petermossamlallresearch.com/students/student/taru-jain/profile "Taru Jain") and Adam's [projects](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Keras/AllCNN "projects") based on their work.

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

## Raspberry Pi Buster

For this Project, the operating system choice is [Raspberry Pi Buster](https://www.raspberrypi.org/downloads/raspberry-pi-os/ "Raspberry Pi Buster"), previously known as Raspian. 

## CNN

For this project we will use the model created in the [CNN](CNN "CNN") project. If you would like to train your own model you can follow the CNN guide, or you can use the pre-trained model and weights provided in the **Model** directory.

## Data

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **Model/Data**, inside you have **Test**. 

You need to use the following test data if you are using the pre-trained model, or use the same test data you used when training your own network if you used a different test set there.

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

&nbsp;

# Configuration

[config.json](Model/config.json "config.json")  holds the configuration for our network. 

```
{
    "cnn": {
        "api": {
            "server": "",
            "port": 1234
        },
        "core": [
            "Server",
            "Client",
            "Classify"
        ],
        "data": {
            "dim": 100,
            "file_type": ".jpg",
            "labels": [0, 1],
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
        }
    }
}
```

The cnn object contains 3 Json Objects (api, data and model) and a JSON Array (core). **api** has the information used to set up your server you will need to add your local ip, **data** has the configuration related to preparing the training and validation data, and **model** holds the model file paths. 

&nbsp;

# Local Testing

Now we will use the test data to see how the classifier reacts to our testing data on a Raspberry Pi 4. Real world testing is the most important testing, as it allows you to see the how the model performs in a real world environment. 

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

# HIAS / Rift Classification

To use this project with the [HIAS](https://github.com/LeukemiaAiResearch/HIAS "HIAS") and [Rift](Rift "Rift") classifiers, head to the related links to complete the setup. 

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