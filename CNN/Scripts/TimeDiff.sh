#!/bin/bash

timediff(){
	op=$1
	START=$(date +%s)
	cd ../
	python AllDS2020.py $op
	END=$(date +%s)
	DIFF=$(( $END - $START ))
	printf 'Operation %s took %ddays %dhrs %dmins %dsec\n' $op $(($DIFF/86400)) $(($DIFF%86400/3600)) $(($DIFF%3600/60)) \ $(($DIFF%60))
	cd -
	exit
}

if [[ "$2" == "Classify" ]] || [[ "$2" == "Train" ]] ; then
	timediff $2
else
	echo "Wrong Input Provided"
	echo "Use -op Train or -op Classify"
	exit 0
fi