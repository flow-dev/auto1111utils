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
import math
from datetime import datetime

# タイル画像の基準とする画像サイズ
HEIGHT = 512
WIDTH  = 512

# 最終出力のキャンバスサイズ(正方形のみ)
CANVAS_PIX = 2160


# 画像の中央を切り出す
def crop_image_center(image):

    if (image is None):
        return np.inf
    
    height = int(image.shape[0])
    width = int(image.shape[1])

    # 正方形でない場合は中央をCropする
    if(width != height):
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        image = image[top:bottom, left:right]

    return (image)


# 画像を調整しながら読み込み
def load_images(image_path, scale=1.0):

    image = cv2.imread(image_path)

    # 正方形でない場合は中央をCropする
    image = crop_image_center(image)

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


def paste_image_on_canvas(canvas_size, paste_image):

    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

    # 正方形でない場合は中央をCropする
    paste_image = crop_image_center(paste_image)

    paste_image = cv2.resize(paste_image, (int(canvas_size[1]), int(canvas_size[0])))

    # 画像がキャンバスより小さい場合の余白計算
    y_offset = (canvas_size[1] - paste_image.shape[0]) // 2
    x_offset = (canvas_size[0] - paste_image.shape[1]) // 2

    # 画像を中央に貼り付け
    canvas[y_offset:y_offset + paste_image.shape[0], x_offset:x_offset + paste_image.shape[1]] = paste_image

    return canvas


# 画像群をくっつけて表示
def concat_tile_image(image_list, target_image_path, source_piece_num, n_row=10, n_col=10, scale=1.0, create_video=True):

    height = int(HEIGHT * scale)
    width = int(WIDTH * scale)
    concat_list = []
    canvas_size = (CANVAS_PIX,CANVAS_PIX)
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Target Image Filename
    target_img_filename, extension = os.path.splitext(os.path.basename(target_image_path))
    
    # Define video writer if create_video is True
    if create_video:
        output_file = f"mosic_process_{timestamp}_{target_img_filename}_{str(source_piece_num)}_tiles.mov" # Video file name for saving
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = int(math.ceil(len(image_list) / 5))
        # Ensure fps is within the specified range [1, 24]
        fps = max(1, min(24, fps))
        # print("fps=",fps)
        video_output = cv2.VideoWriter(output_file, fourcc, fps, (CANVAS_PIX, CANVAS_PIX))
        # Display text on the first frame
        image_char = np.ones((CANVAS_PIX, CANVAS_PIX, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 2
        text_lines = ["Generative Mosaic Art - Made with Code", f"Mosaics created from {str(source_piece_num)} images, each {str(n_row)}x{str(n_col)} pixels"]
        line_height = max(cv2.getTextSize("A", font, font_scale, font_thickness)[0][1], 10)  # Get the height of the text
        for i, text_line in enumerate(text_lines):
            text_size = cv2.getTextSize(text_line, font, font_scale, font_thickness)[0]
            text_position = ((CANVAS_PIX - text_size[0]) // 2, int((CANVAS_PIX - len(text_lines) * line_height) / 2) + (i + 1) * (line_height+25))
            text_color = (0, 0, 0) #black
            cv2.putText(image_char, text_line, text_position, font, font_scale, text_color, font_thickness)
        for _ in range(48):# Repeat for 48 frames to pause the video
            video_output.write(image_char)

    for idx, image in tqdm(enumerate(image_list), total=len(image_list), desc="[Concat tile images]"):

        if idx % n_col == 0:
            row_list = [image]
            #show_img(np.hstack(row_list)) #左端のタイル
            # if create_video:
            #     print("Write Video Frame idx:",idx)
            #     frame = paste_image_on_canvas(canvas_size, np.hstack(row_list))
            #     video_output.write(frame)
        else:
            row_list = row_list + [image]
            #show_img(np.hstack(row_list)) #左端から積まれていくタイル
            # if create_video:
            #     print("Write Video Frame idx:",idx)
            #     frame = paste_image_on_canvas(canvas_size, np.hstack(row_list))
            #     video_output.write(frame)
            if idx % n_col == (n_col -1):
                row_img = np.hstack(row_list)
                concat_list = concat_list +[row_img]
                #show_img(np.vstack(concat_list)) #縦に積まれたタイル
                if create_video:
                    # 偶数のみフレーム書き出しで拡大表示時の端画像をきれいにする
                    # 5x5ブロックから表示する制約を入れる。3x3までは拡大のジャギーが目立つため
                    if((idx % 2 == 0) & (idx >= ((n_col * 5)-1))):
                        #print("Write Video Frame idx:",idx,n_col)
                        frame = paste_image_on_canvas(canvas_size, np.vstack(concat_list))
                        for _ in range(7):#7フレーム繰り返し書いて映像を止める
                            video_output.write(frame)
        if idx == len(image_list) - 1:
            if idx % n_col < n_col -1:
                n_blank = n_col - (len(image_list) % n_col)
                row_list = row_list + [np.zeros( (width, height, 3) )] * n_blank
                row_img = np.hstack(row_list)
                concat_list = concat_list + [row_img]
        if idx >= n_row * n_col:
            break

    #最終タイル画像の生成
    concat_img = np.vstack(concat_list)
    
    if create_video:

        # #白画像を挟んでフラッシュ効果を表示
        # canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        # frame = paste_image_on_canvas(canvas_size, canvas)
        # for _ in range(6):#6フレーム繰り返し書いて映像を止める
        #     video_output.write(frame)

        "完成したモザイク画像を動画に表示"
        frame = paste_image_on_canvas(canvas_size, concat_img)
        for _ in range(96):#96フレーム繰り返し書いて映像を止める
            video_output.write(frame)
        
        "モザイク前の目標画像を動画に表示"
        target_image = cv2.imread(target_image_path)

        # 正方形でない場合は中央をCropする
        target_image = crop_image_center(target_image)

        target_image = cv2.resize(target_image, (concat_img.shape[1], concat_img.shape[0]))
        frame = paste_image_on_canvas(canvas_size, target_image)
        for _ in range(72):#72フレーム繰り返し書いて映像を止める
            video_output.write(frame)

    # 画像を保存する
    resized_image = cv2.resize(concat_img, (CANVAS_PIX, CANVAS_PIX))
    concat_img_name = f"concat_img_{timestamp}_{target_img_filename}_{str(source_piece_num)}_tiles.jpg"
    cv2.imwrite(concat_img_name, resized_image)

    # Release the video writer if create_video is True
    if create_video:
        video_output.release()

    print("Finish Generate Image & Video")

    # 画像を表示
    # show_img(concat_img)

    return

def mosaic(input_dir, target_image_path, mode="LAB2", n_div=110, piece_scale=1/40):

    n_row = n_div
    n_col = n_div
    eval_height = 16
    eval_width = 16
    
    "[1] 並べる画像を読み込み"
    # モザイクアートに使う写真をリサイズして読み込み
    image_names = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg"))])
    source_piece_list = [load_images(os.path.join(input_dir, p),scale=piece_scale) for p in tqdm(image_names, desc="[Load source images]")]

    # ソースとなるピース数を保持
    source_piece_num = len(source_piece_list)
    
    # 比較用にリサイズしたものも準備
    source_eval_piece_list = [cv2.resize(img, (eval_width, eval_height), interpolation=cv2.INTER_AREA) for img in tqdm(source_piece_list,desc="[Resize source images]")]
    
    "[2] 目的の画像を読み込み"
    target_image = cv2.imread(target_image_path)

    # 正方形でない場合は中央をCropする
    target_image = crop_image_center(target_image)

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
    concat_tile_image(mosaic_list_sorted, target_image_path, source_piece_num, n_col=n_col, n_row=n_row, scale=piece_scale)

if __name__ == '__main__':
    input_dir = "./" ### モザイクアートに使う写真が格納されているディレクトリを指定
    target_image_path = "./00030-3053354251.png" ### モザイクアートで作りたい画像のパスを指定

    mode = "LAB" ### 利用する比較モードを指定
    n_div = 63 ### 作りたい画像の各辺を何分割するかを指定(n_div * n_div枚をモザイクアートで使うことになる)
    piece_scale = 1 / 15 ### モザイクアートに並べる画像のサイズの倍率を指定

    mosaic(input_dir, target_image_path, mode=mode, n_div=n_div, piece_scale=piece_scale)
