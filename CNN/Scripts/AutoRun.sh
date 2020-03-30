#!/bin/bash

echo "Installing dependencies via Setup Script"
sh ../Setup.sh
read -p 'Do you want to randomize the image transfer in Test folder? (Y/N) - ' input
read -p 'Enter path to im folder of ALL_IDB1 (in Windows path should be in format C:/Users/ instead of C:\Users\) - ' imPath
if [[ "$input" == "Y" ]]; then
	./Randomizer.sh -path $imPath -randomize $input
else
	./Randomizer.sh -path $imPath -randomize N
fi
echo "-- Starting training"
echo " "
cd -
TimeDiff.sh -op Train
echo "-- Finished training"
echo "-- Starting classifying"
echo " "
TimeDiff.sh -op Classify
echo "-- Finished classifying. Exiting Script."