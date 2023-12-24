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

def create_image_grid():

    # Read all png files
    image_files = glob.glob("*.png")

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


    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Gen filepath
    output_path = f"output_grid_{timestamp}_{str(num_images)}_images.jpg"

    # Save a grid image
    output_image = Image.fromarray(grid_image)
    output_image = output_image.resize((1080, 1080))
    output_image.save(output_path)
    print(f"Grid image saved at: {output_path}")

if __name__ == '__main__':
    create_image_grid()
