#!/bin/bash

img_list="config/GA_S1_image_list.txt"
n_img=`cat ${img_list} | wc -l`
counter=0

echo " "
echo "Checking gamma to sigma conversion for all images in img_list"
echo "Image list has ${n_img} entries"

for f in `cat ${img_list}`; do

    # Increase counter
    let counter+=1

    echo " "
    echo "Processing image ${counter}/${n_img} "
    echo "Processing ${f}"
    echo " "

    conda run -n SWOT_analysis python gamma_2_sigma_conversion_check.py ${f}

done 
