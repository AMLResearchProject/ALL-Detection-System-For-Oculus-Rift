#!/bin/sh

echo "Copying scripts to their respective directories"
echo " "
echo "Copying Autorun.sh, randomizer.sh and timediff.sh scripts to the CNN directory"
cp Autorun.sh ../CNN
cp timediff.sh ../CNN
cp randomizer.sh ../CNN/Model/Data/Train/
echo "Done"
exit
