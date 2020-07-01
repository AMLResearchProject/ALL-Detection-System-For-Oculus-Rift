# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

# Anaconda Installation

Anaconda lets you create virtual environments for your projects. Within the environments you can install Python packages that will not interfere with your core Python installations. A particularly useful feature is that you are able to spin up an environment with Tensorflow GPU in a few seconds, this installation is separate from you system installation meaning you can play around without causing issues. 

&nbsp;

## Installation

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
source ~/.bashrc
```

Now see if it has worked:

```
conda list
```
If you see a list of packages, conda is working. 

## Using Anaconda

A good way to use Anaconda is create a directory on your desktop, and inside there have a directory for every version of Tensorflow you want to use. If you are using GPU you could have the a folder for CPU and GPU versions also.

- Environments
    - TF2
        - TF2 CPU
        - TF2 GPU

To create a CPU installation of Tensorflow, navigate to your TF2 CPU directory and use the following command:

```
conda create --name tf_2 python=3 tensorflow
```

To create a GPU installation of Tensorflow, navigate to your TF2 GPU directory and use the following command:

```
conda create --name tf_2_gpu python=3 tensorflow-gpu
```
To create an environment with an older version of Tensorflow, you can use the following command, replacing the version number with your required version number. This works for both CPU and GPU versions.

```
conda create --name tf_2_gpu python=3 tensorflow-gpu=1.15.0
```
You can activate a conda environment by using the following command and the relevant env name:

```
conda activate tf_2_gpu
```
And you can deactivate an environment using the following:

```
conda deactivate
```

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/team/adam-milton-barker "Adam Milton-Barker") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") President/Lead Developer, Sabadell, Spain

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.