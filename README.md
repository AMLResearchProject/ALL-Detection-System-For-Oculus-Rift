# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

[![CURRENT RELEASE](https://img.shields.io/badge/CURRENT%20RELEASE-0.2.0-blue.svg)](https://github.com/AMLResearchProject/ALL-Detection-System-2020/tree/0.2.0) [![UPCOMING RELEASE](https://img.shields.io/badge/CURRENT%20DEV%20BRANCH-0.3.0-blue.svg)](https://github.com/AMLResearchProject/ALL-Detection-System-2020/tree/0.3.0)

![Acute Lymphoblastic Leukemia Detection System 2020](Media/Images/AllVrCNN.jpg)

&nbsp;

# Table Of Contents

- [Introduction](#introduction)
- [DISCLAIMER](#disclaimer)
- [ALL-IDB](#all-idb)
  - [ALL_IDB1](#all_idb1)
- [Classifier](#classifier)
  - [Results Overview](#results-overview)
- [Setup](#classifier-results-overview)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;

# Introduction

The Acute Lymphoblastic Leukemia Detection System 2020 uses Tensorflow 2 & Oculus Rift, and Raspberry Pi 4 to provide a virtual diagnosis system.

This project is made up of a number of components which work together as a network to make the system work. Follow the completed tutorials below in the provided order. The classifiers can be run indvidually but a full system setup requires the [Tensorflow 2.0 CNN](CNN "Tensorflow 2.0 CNN"), [Virtual Reality (Oculus Rift)](Rift "Oculus Rift"), an [RPI 4](RPI4 "RPI 4") and [Arduino IoT Alarm](Arduino "Arduino IoT Alarm").

**This project is compatible with the Acute Lymphoblastic Detection System feature of [HIAS](https://github.com/LeukemiaAiResearch/HIAS "HIAS").**

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

# Classifier

<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In this project we will a network based on the proposed architecture in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper, built using Tensorflow 2.

## Classifier Results

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

# Setup

Follow the guides below to build your Virtual Reality Acute Lymphoblastic Leukemia Detection System. The classifiers can be used as standalone if required, but this project focuses on the system as a whole.

| Project                                                                                                                                                                                                                                                                                                                                                                      | Description                                                                                                                         | Status      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [Rift](Rift "Rift")                                                                                                                                                                                                                                                             | A virtual room made with Unreal Engine 4. In the room the ALL data is displayed as blocks and the user can point at those blocks to classify the image.                      | Development    |
| [CNN](CNN "CNN") | Applies filters to the original dataset and increases the amount of training / test data. Provides code for training a CNN for detecting ALL. Hosts a REST API endpoint that provides access to the model for inference. | COMPLETE    |
| [RPI 4](RPI4 "RPI 4") | Applies filters to the original dataset and increases the amount of training / test data. Provides code for training a CNN for detecting ALL. Hosts a REST API endpoint that provides access to the model for inference. | COMPLETE    |
| [Arduino](https://github.com/AMLResearchProject/ALL-Detection-System-2020/tree/master/Arduino "Arduino")                                                                                                                                                                                                                                                             | An IoT connected Arduino ioT Alarm using an Arduino UNO, an ESP8266 and the [HIAS](https://github.com/LeukemiaAiResearch/HIAS "HIAS") iotJumpWay MQTT broker.                      | Development    |

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

- [Javier Lopez Alonso](https://www.leukemiaresearchassociation.ai/team/javier-lopez-alonso "Javier Lopez Alonso") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Treasurer/Research & Development, Sabadell, Spain

- [Rishabh Banga](https://www.leukemiaresearchassociation.ai/team/rishabh-banga "Rishabh Banga") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Research & Development, Delhi, India

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE.md "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.