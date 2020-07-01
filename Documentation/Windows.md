# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

&nbsp;

**Issue #1
ImportError: DLL load failed: The specified module could not be found**

**OS -** Windows 10

**Reason -** As per TF's Release Notes for 2.1.0. Officially-released Tensorflow Pip packages are now built with Visual Studio 2019 version 16.4 in order to take advantage of the new /d2ReducedOptimizeHugeFunctions compiler flag. 

To use these new packages, you must install "Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019", available from Microsoft's website here.

**Solution -** If you are facing this after Tensorflow 2.1.0, it is probably because Tensorflow (CPU version) comes with GPU support by default. And it requires Microsoft Visual C++ Redistributable for Visual Studio 2015/17/19 as shown in the installation step #1 on the website.

**Issue #2 - Win10: AttributeError: 'RepeatedCompositeFieldContainer' object has no attribute 'append'**

**OS -** Windows 10

**Reason -** For binaries, the PIP dependency is protobuf >= 3.9.2. The same can be referred from [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py#L64)

**Solution -** Run pip install -U protobuf to install latest version of protobuf

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

&nbsp;

## Contributors

- [Rishabh Banga](https://www.leukemiaresearchassociation.ai/team/rishabh-banga "Rishabh Banga") - [Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss](https://www.leukemiaresearchassociation.ai "Asociacion De Investigation En Inteligencia Artificial Para La Leucemia Peter Moss") Research & Development, Delhi, India

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](../releases "Releases").

&nbsp;

# License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE "LICENSE") file for details.

&nbsp;

# Bugs/Issues

We use the [repo issues](../issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](../CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.