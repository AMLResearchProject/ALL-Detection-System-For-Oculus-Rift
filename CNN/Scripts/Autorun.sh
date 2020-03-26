#!/bin/bash

echo "Running Dependencies Setup Script"
sh setup.sh
read -p 'Do you want to randomize the image transfer in Test folder? (Y/N) - ' input
read -p 'Enter path to im folder of ALL_IDB1 (in Windows path should be in format C:/Users/ instead of C:\Users\) - ' imPath
cd Model/Data/Train/
if [[ "$input" == "Y" ]]; then
	./randomizer.sh -path $imPath -randomize $input
else
	./randomizer.sh -path $imPath -randomize N
fi
echo "-- Starting training"
echo " "
cd -
timediff.sh -op Train
echo "-- Finished training"
echo "-- Starting classifying"
echo " "
timediff.sh -op Classify
echo "-- Finished classifying. Exiting Script."