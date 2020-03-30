#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
echo "-- Installing requirements"
echo " "
conda install opencv
conda install requests
conda install flask
conda install matplotlib
conda install numpy
conda install scipy
conda install Pillow
conda install jsonpickle
conda install scikit-learn
conda install scikit-image
echo "-- Done"