#!/bin/bash

# Set the paths for the input and output folders
input_folder="/stable-diffusion-webui/outputs/car_mosaic"
output_folder="/stable-diffusion-webui/outputs/auto_mosaic"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Process all .mov files in the input folder
for mov_file in "$input_folder"/*.mov; do
    # Extract the file name without extension and generate a new file name
    filename=$(basename "$mov_file")
    filename_noext="${filename%.*}"
    output_path="$output_folder/resized_${filename_noext}.mov"

    # Use ffmpeg to resize the video
    ffmpeg -i "$mov_file" -vf "scale=1080:1080" "$output_path"
done

echo "Resizing completed."
