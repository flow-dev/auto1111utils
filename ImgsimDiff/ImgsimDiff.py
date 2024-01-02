#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ImgsimDiff.py
"""

import cv2
import imgsim
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

def calculate_image_distance(image_path1, image_path2):
    img0 = cv2.imread(image_path1)
    img1 = cv2.imread(image_path2)

    vtr = imgsim.Vectorizer()
    vec0 = vtr.vectorize(img0)
    vec1 = vtr.vectorize(img1)

    dist = imgsim.distance(vec0, vec1)

    print("Distance between the images:", dist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image distance based on vectorization.")
    parser.add_argument("image_path1", help="Path to the first image.")
    parser.add_argument("image_path2", help="Path to the second image.")
    args = parser.parse_args()

    calculate_image_distance(args.image_path1, args.image_path2)
