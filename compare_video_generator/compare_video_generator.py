#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
compare_video_generator.py
"""

import os
import numpy as np
import glob
import cv2
from datetime import datetime
import math
from moviepy.editor import VideoFileClip, AudioFileClip
import librosa


# 画像の中央を切り出す
def crop_image_center(image):

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
    
    image = cv2.resize(image, (512, 512))

    return (image)


def write_text_image(image, text, text_flag=False):

    if(text_flag):
        # 下部にテキストを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, image.shape[0] - 10)
        font_scale = 1.0
        font_color = (255, 255, 255)
        line_type = 1

        cv2.putText(image, text, bottom_left_corner, font, font_scale, font_color, line_type)

    return image


def paste_image_on_canvas(canvas_size, paste_image, text_to_add, text_flag=False):

    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    h, w = paste_image.shape[:2]
    target_w, target_h = canvas_size

    aspect_ratio_orig = w / h
    aspect_ratio_target = target_w / target_h

    # ターゲットサイズに合わせてリサイズ
    if aspect_ratio_orig > aspect_ratio_target:
        # 元画像のアスペクト比が大きい場合（横に長い場合）
        new_w = target_w
        new_h = int(target_w / aspect_ratio_orig)
    else:
        # 元画像のアスペクト比が小さい場合（縦に長い場合）
        new_h = target_h
        new_w = int(target_h * aspect_ratio_orig)

    paste_image = cv2.resize(paste_image, (new_w, new_h))

    # 画像がキャンバスより小さい場合の余白計算
    y_offset = (canvas_size[1] - paste_image.shape[0]) // 2
    x_offset = (canvas_size[0] - paste_image.shape[1]) // 2

    # 画像を中央に貼り付け
    canvas[y_offset:y_offset + paste_image.shape[0], x_offset:x_offset + paste_image.shape[1]] = paste_image

    if(text_flag):
        # テキストを上段中央に追記
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text_to_add, text_font, 1, 2)[0]
        text_position = ((canvas_size[0] - text_size[0]) // 2, int(text_size[1] * 1.5))
        font_scale = 1.0
        font_color = (255, 255, 255)
        line_type = 2
        cv2.putText(canvas, text_to_add, text_position, text_font, font_scale, font_color, line_type)

    return canvas


def calculate_frames_per_beat(bpm, fps):
    "https://note.com/oshibataku/n/nbaa5a2da1bca"
    beat_per_seconds = bpm / 60
    frames_per_beat = fps / beat_per_seconds
    return int(frames_per_beat-2)


if __name__ == '__main__':

    CANVAS_SIZE = (540, 960)
    FPS = 24
    MOV_TIMES = 20.0

    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"compare_video_{timestamp}.mov" # Video file name for saving
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    video_output = cv2.VideoWriter(output_file, fourcc, FPS, (CANVAS_SIZE[0], CANVAS_SIZE[1]))

    # Load the audio file
    audio_file = "MooveKa - Busy Day Ahead.mp3"
    audio = AudioFileClip(audio_file)

    y, sr = librosa.load(audio_file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    frames_per_beat = calculate_frames_per_beat(tempo,FPS)
    print("BPM & frames_per_beat", tempo, frames_per_beat)


    before_dir = "images/"
    after_dir = "images/"

    before_file_list = glob.glob(os.path.join(before_dir, '*.png'))
    after_file_list = glob.glob(os.path.join(after_dir, '*.png'))

    before_file_num = len(before_file_list)
    after_file_num = len(after_file_list)

    for i in range(min(before_file_num, after_file_num)):
        before_image_path = before_file_list[i]
        after_image_path = after_file_list[i]

        output_image_path = f"merged_image_{i+1}.jpg"

        before_image = cv2.imread(before_image_path)
        after_image = cv2.imread(after_image_path)

        before_image = write_text_image(crop_image_center(before_image), "before")
        after_image = write_text_image(crop_image_center(after_image), "after")

        merged_image = before_image
        #merged_image = cv2.hconcat([before_image, after_image])
        #merged_image = cv2.vconcat([before_image, after_image])


        merged_image = paste_image_on_canvas(CANVAS_SIZE, merged_image, "Deep White Balance(AWB)")
        for _ in range(frames_per_beat):# Repeat for fps*1 frames to pause the video
            video_output.write(merged_image)

        # cv2.imshow('Crop Image', merged_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Release the video writer
    video_output.release()

    # Load the video without audio using moviepy
    video = VideoFileClip(output_file)

    # Trim or loop the audio to match the video duration
    audio = audio.subclip(0, video.duration)

    # Set the audio for the video
    video = video.set_audio(audio)

    # Write the final video with audio
    final_output_file = f"compare_video_with_audio_{timestamp}.mov"
    video.write_videofile(final_output_file, codec='libx264', audio_codec='aac')

    # Close video file
    video.close()


