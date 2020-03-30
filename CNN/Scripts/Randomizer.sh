#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "\n"
    printf "Options:\n"
    printf " -path, Enter path to im folder of ALL_IDB1 (If Windows use full path starting with /mnt/c/\n"
    printf " -randomize Y, Randomizes the images being transferred to Test Folder. Only use if not performance testing\n" 
    printf "  i.e. Recreating the exact scenario using fixed test files\n"
    printf " -h, Print help\n"
return 0
}

function incorrect_usage (){
	echo "-- Incorrect input provided"
	echo " "
	show_usage
	echo " "
	echo "Exiting script --"
	exit
}

if [[ "$1" == "-h" ]]; then
	show_usage
	exit
elif [[ "$1" == "-path" ]]; then
	imPath=$2
	echo "Copying images to current path"
	cp $imPath/* ../Model/Data/Train/
	if [[ "$4" == "Y" ]]; then
		echo "Copying random images to Test folder"
		shuf -zn10 -e ../Model/Data/Train/*_1.jpg | xargs -0 mv -vt ../Model/Data/Test/
		shuf -zn10 -e ../Model/Data/Train/*_0.jpg | xargs -0 mv -vt ../Model/Data/Test/
	elif [[ "$4" == "N" ]]; then
		mv im006_1.jpg im020_1.jpg im024_1.jpg im026_1.jpg im028_1.jpg im031_1.jpg im035_0.jpg im041_0.jpg im047_0.jpg im053_1.jpg im057_1.jpg im060_1.jpg im063_1.jpg im069_0.jpg im074_0.jpg im088_0.jpg im095_0.jpg im099_0.jpg im101_0.jpg im106_0.jpg ../Test/
	else
		incorrect_usage
	fi
	echo "Done"
else
	incorrect_usage
	exit
fi