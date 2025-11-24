#!/bin/bash

img_list="config/S1_image_list.txt"
n_img=`cat ${img_list} | wc -l`
counter=0

echo " "
echo "Geocoding all scenes from image list"
echo "Image list has ${n_img} entries"

for f in `cat ${img_list}`; do

    # Increase counter
    let counter+=1

    echo " "
    echo "Processing image ${counter}/${n_img} "
    echo "Processing ${f}"
    echo " "

    conda run -n geocoding python S1_geocode_image.py ${f}

done 
