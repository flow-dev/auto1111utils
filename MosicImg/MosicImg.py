#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
MosicImg.py
make mosic art from many images
https://note.com/lucax2/n/n67cf9b6a469e
"""

__version__ = "1.00"
__date__ = "25 Dec 2023"

import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# タイル画像の基準とする画像サイズ
HEIGHT = 512
WIDTH  = 512


# 画像を調整しながら読み込み
def load_images(image_path, scale=1.0):

    image = cv2.imread(image_path)

    height = int(image.shape[0])
    width = int(image.shape[1])

    # 正方形でない場合は中央をCropする
    if(width != height):
        min_dim = min(width, height)
        print("Crop the image from center=", image_path)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image[top:bottom, left:right]

    # タイル画像の基準とする画像サイズにリサイズする
    image = cv2.resize(image, (int(WIDTH*scale), int(HEIGHT*scale)), interpolation=cv2.INTER_AREA)
    return image


# 画像を分割する
def divide_image(img, n_row=4, n_col=4):
    height, width, _ = np.shape(img)
    divide_list = [[img[int(row_ind * (height / n_row)) : int((row_ind + 1) * (height /n_row)), 
                                    int(col_ind *(width / n_col)) : int((col_ind + 1) * (width  / n_col)), :]
                            for col_ind in range(n_col)] for row_ind in range(n_row)]
    return divide_list


# 2つの画像の間のスコアを計算
def diff_score(img1, img2):
    if (img1 is None) or (img2 is None):
        return np.inf
    
    diff_score = np.sum(np.sqrt(np.sum((img1 - img2)**2, axis=2)))
    return diff_score


# 2つの画像の間のスコアを計算(Hueの計算用)
def diff_score_hsv(img1, img2, color_coef=np.array([1, 1, 1])):
    if (img1 is None) or (img2 is None):
        return np.inf
    
    #Hueに関する誤差の計算
    max_hue = 180 * color_coef[0]
    diff_hue = np.abs(img1[:,:,0] - img2[:,:,0])
    diff_hue[diff_hue > (max_hue * 0.5)] = max_hue - diff_hue[diff_hue > (max_hue * 0.5)]

    #他の部分の誤差とあわせる
    diff_score = np.sum(np.sqrt(np.sum((np.dstack([diff_hue, img1[:,:,1:] - img2[:,:,1:]]))**2, axis=2)))
    return diff_score


# 画像を調整しながら表示
def show_img(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)#, cmap = 'gray')
    #plt.colorbar()
    plt.show()


# 画像群をくっつけて表示
def concat_tile_image(image_list, n_row=10, n_col=10, scale=1.0):

    height = int(HEIGHT * scale)
    width = int(WIDTH * scale)
    concat_list = []

    for idx, image in enumerate(image_list):
        if idx % n_col == 0:
            row_list = [image]
        else:
            row_list = row_list + [image]
            if idx % n_col == (n_col -1):
                row_img = np.hstack(row_list)
                concat_list = concat_list +[row_img]
        if idx == len(image_list) - 1:
            if idx % n_col < n_col -1:
                n_blank = n_col - (len(image_list) % n_col)
                row_list = row_list + [np.zeros( (width, height, 3) )] * n_blank
                row_img = np.hstack(row_list)
                concat_list = concat_list + [row_img]            
        if idx >= n_row * n_col:
            break
    concat_img = np.vstack(concat_list)

    # 画像を表示
    show_img(concat_img)

    resized_image = cv2.resize(concat_img, (1080, 1080))
    cv2.imwrite('concat_img.jpg', resized_image)

def mosaic(input_dir, target_image_path, mode="LAB2", n_div=110, piece_scale=1/40):

    n_row = n_div
    n_col = n_div
    eval_height = 16
    eval_width = 16
    
    "[1] 並べる画像を読み込み"
    # モザイクアートに使う写真をリサイズして読み込み
    image_names = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg"))])
    source_piece_list = [load_images(os.path.join(input_dir, p),scale=piece_scale) for p in tqdm(image_names, desc="[Load source images]")]
    
    # 比較用にリサイズしたものも準備
    source_eval_piece_list = [cv2.resize(img, (eval_width, eval_height), interpolation=cv2.INTER_AREA) for img in tqdm(source_piece_list,desc="[Resize source images]")]
    
    "[2] 目的の画像を読み込み"
    target_image = cv2.imread(target_image_path)
    height = int(target_image.shape[0])
    width = int(target_image.shape[1])

    # 正方形でない場合は中央をCropする
    if(width != height):
        min_dim = min(width, height)
        print("Crop the image from center=", target_image_path)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        target_image = target_image[top:bottom, left:right]

    "[3] 目的の画像を分割"
    target_piece_list = divide_image(target_image, n_col=n_col, n_row=n_row)
    
    "[4] 分割した画像のピースをシャッフル"
    # ピースをランダムな順番に並べ替える(後から復元可能にする)
    tile_ind_list = [(r_ind, c_ind) for c_ind in range(n_col) for r_ind in range(n_row)]
    tile_ind_list = random.sample(tile_ind_list, k=len(tile_ind_list))
    rand_target_piece_list = [target_piece_list[ind[0]][ind[1]] for ind in tile_ind_list]

    # 比較用にピースをリサイズ
    rand_target_piece_list = [cv2.resize(img, (eval_width, eval_height), interpolation=cv2.INTER_AREA) for img in tqdm(rand_target_piece_list, desc="[Resize target pieces]")]
    
    "[5] ピースごとに並べる画像と比較して採用画像を決定"
    # 比較モードごとに色空間を変換(係数はお好みで調整)

    if mode == "HSV":
        color_mode = cv2.COLOR_BGR2HSV
        color_coef = np.array([1, 1, 1])
    elif mode == "Hue":
        color_mode = cv2.COLOR_BGR2HSV
        color_coef = np.array([4, 1, 1])
    elif mode == "LAB":
        color_mode = cv2.COLOR_BGR2LAB
        color_coef = np.array([1, 1, 1])
    elif mode == "LAB2":
        color_mode = cv2.COLOR_BGR2LAB
        color_coef = np.array([2, 1, 1])
    elif mode == "LAB4":
        color_mode = cv2.COLOR_BGR2LAB
        color_coef = np.array([4, 1, 1])
    
    source_eval_piece_list = [cv2.cvtColor(img, color_mode) * color_coef for img in source_eval_piece_list]
    rand_target_piece_list = [cv2.cvtColor(img, color_mode) * color_coef for img in rand_target_piece_list]

    # ピースごとにスコアを比較
    mosaic_list = []
    for target_piece in tqdm(rand_target_piece_list, desc="[Score matching]"):
        if mode in ["HSV", "Hue"]:
            score = np.array([diff_score_hsv(target_piece, s, color_coef=color_coef) for s in source_eval_piece_list])
        else:
            score = np.array([diff_score(target_piece, s) for s in source_eval_piece_list])
        min_ind = np.argmin(score)
    
        # 採用画像をモザイクアートリストに加えて、元のプールから削除
        mosaic_list += [source_piece_list[min_ind]]
        source_eval_piece_list[min_ind] = None
    
    "[6] シャッフル順を戻す"
    # モザイクアートリストをピースをランダムに並べ替えた時の元の順番に並べ替える
    tile_argind_list = np.argsort([ind[0] * n_col + ind[1] for ind in tile_ind_list])
    mosaic_list_sorted = [mosaic_list[ind] for ind in tile_argind_list]
    
    "[7] くっつけて一つのモザイクアートにする"
    # 並べ替えたピースを一つの画像にくっつけて表示
    concat_tile_image(mosaic_list_sorted, n_col=n_col, n_row=n_row, scale=piece_scale)

if __name__ == '__main__':
    input_dir = "./" ### モザイクアートに使う写真が格納されているディレクトリを指定
    target_image_path = "./001.png" ### モザイクアートで作りたい画像のパスを指定

    mode = "LAB" ### 利用する比較モードを指定
    n_div = 50 ### 作りたい画像の各辺を何分割するかを指定(n_div * n_div枚をモザイクアートで使うことになる)
    piece_scale = 1 / 20 ### モザイクアートに並べる画像のサイズの倍率を指定

    mosaic(input_dir, target_image_path, mode=mode, n_div=n_div, piece_scale=piece_scale)
