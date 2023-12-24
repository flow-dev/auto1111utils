#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
MakeGridImg.py
make grid for many image result
"""

__version__ = "1.00"
__date__ = "24 Dec 2023"

import glob
import numpy as np
from PIL import Image
from datetime import datetime
import random
import cv2
import math

def create_image_grid(sorted_flag=False, random_flag=True, create_video=True):

    # Read all png files
    image_files = glob.glob("*.png")
    
    # Random shuffle
    if(random_flag):
        random.shuffle(image_files)

    if(sorted_flag):
        image_files = sorted(image_files)

    # Calc images number
    num_images = len(image_files)

    # Calc grid size
    grid_size = int((np.sqrt(num_images)))
    print("grid_size=",grid_size)

    # Calc grid size and images number diff
    grid_diff = int(num_images - (grid_size*grid_size))

    # del the image list
    if(grid_diff != 0):
        print("adjust grid diff ofset=",grid_diff)
        image_files = image_files[:-grid_diff]
        num_images = len(image_files)
    
    print("num_images=",num_images)

    # init grid canvas
    grid_image = np.ones((grid_size * 512, grid_size * 512, 3), dtype=np.uint8) * 255

    # Define video writer if create_video is True
    if create_video:
        output_file = "grid_generation_process.mov"  # 保存する動画ファイル名
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = int(math.ceil(num_images / 5))
        # Ensure fps is within the specified range [1, 24]
        fps = max(1, min(24, fps))
        print("fps=",fps)
        video_output = cv2.VideoWriter(output_file, fourcc, fps, (1080, 1080))

    # put the images on the grid canvas
    for i, image_path in enumerate(image_files):
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size
        min_dim = min(width, height)
        if(width != height):
            print("Crop the image from center=",i, image_path)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = (width + min_dim) // 2
            bottom = (height + min_dim) // 2
            img = img.crop((left, top, right, bottom)) 
        img = img.resize((512, 512))
        
        row = i // grid_size
        col = i % grid_size

        grid_image[row * 512:(row + 1) * 512, col * 512:(col + 1) * 512, :] = np.array(img)

        # Add the current frame to the video if create_video is True
        if create_video:
            frame = cv2.resize(grid_image, (1080, 1080))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # output_image = Image.fromarray(frame)
            # process_path = f"output_{str(i)}_images.jpg"
            # output_image.save(process_path)
            video_output.write(frame)

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Gen filepath
    output_path = f"output_grid_{timestamp}_{str(num_images)}_images.jpg"

    # Save a grid image
    output_image = Image.fromarray(grid_image)
    output_image = output_image.resize((1080, 1080))
    output_image.save(output_path)
    print(f"Grid image saved at: {output_path}")

    # Release the video writer if create_video is True
    if create_video:
        video_output.release()

if __name__ == '__main__':
    create_image_grid()
