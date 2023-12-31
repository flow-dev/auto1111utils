#!/bin/bash

# Set the paths for the input and output folders
input_folder="/stable-diffusion-webui/outputs/MosaicArt/camera_mosaic"
output_folder="/table-diffusion-webui/outputs/MosaicArt/target_images"

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

# Move all jpg files from jpg_input_folder to jpg_output_folder
mv "$input_folder"/*.jpg "$output_folder/"

echo "Resizing and Move all jpg files completed."