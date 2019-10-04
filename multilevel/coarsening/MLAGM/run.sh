#!/bin/bash

# example parameter
#dp=./graphs/3elt.graph.levels/     
#gf=3elt
#t=10
#r=1
dp=$1; gf=$2; t=$3; r=$4

# clean existing temp file in the data path
find $dp -name "mat_l*" | xargs -I {} rm {}

# convert the text graph to adjacency matrix in PETSc format
python readTextGraphFiles.py -d $dp -f $gf".graph"

# run the coarsening and store the required matrices
./mlamg --ds_p $dp -f $gf -t $t -r $r

# find the number of levels 
maxlevel=`ls $dp | grep mat_l | grep -v info | awk -F'_' '{print $3}' | sort -r | head -1 `
echo "maxlevel:"$maxlevel

# generate text graph files and strength files
for i in `seq $maxlevel`; do python writeMatrix2Text.py -d $dp -l $i  ; done

echo "The "$maxlevel" level of coarsening are done."
