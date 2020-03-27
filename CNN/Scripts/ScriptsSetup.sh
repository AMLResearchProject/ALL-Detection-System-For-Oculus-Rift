#!/bin/sh

echo "Copying scripts to their respective directories"
echo " "
echo "Copying Autorun.sh, Randomizer.sh and Timediff.sh scripts to the CNN directory"
cp Autorun.sh ../
cp Timediff.sh ../
cp Randomizer.sh ../Model/Data/Train/
echo "Done"
echo " "
cd ../
echo "Activating AutoPilot mode"
sh Autorun.sh
echo " "
exit