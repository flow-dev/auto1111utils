#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
DAWB.py
マスターモニタを持ち込む必要がなく、高度なディープラーニングアルゴリズムを活用して最適なホワイトバランス（WB）設定値を数値で推定します。
Deep White Balanceは正確な色表現を提供し、AugNetを用いて目標画像と現在画像の類似度を数値化することで、的確なWB調整を実現を狙います。
このツールはWBキャリブレーションプロセスを効率的かつ手軽に行えるようにすることを目指しています。
"""

import cv2
import torch
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from arch import deep_wb_single_task
from utilities.deepWB import deep_wb
import utilities.utils as utls
import imgsim
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

def load_deep_wb():
    "Load Deep WB models"
    
    model_dir = "./models"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load awb net
    net_awb = deep_wb_single_task.deepWBnet()
    net_awb.to(device=device)
    net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'), map_location=device))
    net_awb.eval()

    # load tungsten net
    net_t = deep_wb_single_task.deepWBnet()
    net_t.to(device=device)
    net_t.load_state_dict(torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
    net_t.eval()

    # load shade net
    net_s = deep_wb_single_task.deepWBnet()
    net_s.to(device=device)
    net_s.load_state_dict(torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
    net_s.eval()

    return device, net_awb, net_t, net_s


def run_deep_wb(img, device,net_awb, net_t, net_s, mode="AWB"):
    "Eval Deep WB"

    out_awb, out_t, out_s = deep_wb(img, task='all', net_awb=net_awb, net_s=net_s, net_t=net_t, device=device)
    out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)

    if mode == 'AWB':
        result = (out_awb * 255).astype(np.uint8)
        result = write_char(result,"AWB")
        result = Image.fromarray(result)
    elif mode == 'Tungsten':    #2850K
        result = (out_t * 255).astype(np.uint8)
        result = write_char(result,"Tungsten:2850K")
        result = Image.fromarray(result)
    elif mode == 'Fluorecent':  #3800K
        result = (out_f * 255).astype(np.uint8)
        result = write_char(result,"Fluorecent:3800K")
        result = Image.fromarray(result)
    elif mode == 'Daylight':    #5500K
        result = (out_d * 255).astype(np.uint8)
        result = write_char(result,"Daylight:5500K")
        result = Image.fromarray(result)
    elif mode == 'Cloudy':      #6500K
        result = (out_c * 255).astype(np.uint8)
        result = write_char(result,"Cloudy:6500K")
        result = Image.fromarray(result)
    elif mode == 'Shade':       #7500K
        result = (out_s * 255).astype(np.uint8)
        result = write_char(result,"Shade:7500K")
        result = Image.fromarray(result)
    else:
        raise ValueError("Invalid mode. Supported modes are: AWB, Tungsten, Fluorecent, Daylight, Cloudy, Shade")

    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    return result


def write_char(result_image,text_to_write,points=4):
    "画像に文字を書き込む"
    text_position = ((result_image.shape[1] - len(text_to_write) * 10) // 2, result_image.shape[0] - 10)
    
    if points >= 4:
        text_color = (255, 255, 255)
        text_thickness = 1
    else:
        text_color = (0, 255, 0)  # 緑の色
        text_thickness = 2
    
    cv2.putText(result_image, text_to_write, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
    return result_image


def calc_imgsim(frame_0, frame_1):
    vtr = imgsim.Vectorizer()
    vec0 = vtr.vectorize(frame_0)
    vec1 = vtr.vectorize(frame_1)
    dist = imgsim.distance(vec0, vec1)
    return dist

def main():
    # UVCデバイスに接続する
    cap = cv2.VideoCapture(0)  # カメラのデバイス番号（通常は0から始まります）

    if not cap.isOpened():
        print("UVCデバイスに接続できませんでした。")
        return
    
    "Load Deep WB models"
    device, net_awb, net_t, net_s = load_deep_wb()

    # Init AWB mode
    ColorTemperatures = ['AWB', 'Tungsten', 'Fluorecent', 'Daylight', 'Cloudy', 'Shade']
    current_mode_index = 0
    current_mode = ColorTemperatures[current_mode_index]

    while True:
        # フレームを取得
        ret, frame = cap.read()

       # 縦のサイズを320にリサイズ（アスペクト比を保持）
        height, width = frame.shape[:2]
        new_height = 320
        aspect_ratio = new_height / height
        new_width = int(width * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # フレームが正常に取得されたかどうかを確認
        if not ret:
            print("フレームを取得できませんでした。")
            break

        frame_pillow = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #frame_pillow = Image.open("./example_images/Timeline 9_01_00_00_00.jpg")
        res_deep_wb = run_deep_wb(frame_pillow, device, net_awb, net_t, net_s, mode=current_mode)
        #frame_pillow = cv2.cvtColor(np.array(frame_pillow), cv2.COLOR_RGB2BGR)

        dist = calc_imgsim(frame, res_deep_wb)

        frame = write_char(frame, f"AugNet Distance:{dist:.3f}",points=dist)

        show_img = cv2.hconcat([frame, res_deep_wb])

        # 取得したフレームを表示
        cv2.imshow('Deep White Blance', show_img)

        # mキーが押されたらモードをトグルで切り替える
        if cv2.waitKey(1) & 0xFF == ord('m'):
            current_mode_index = (current_mode_index + 1) % len(ColorTemperatures)
            current_mode = ColorTemperatures[current_mode_index]
            print(f"Mode Change!:{current_mode}")

        # 'q'キーが押されたらループを抜けてプログラムを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 使用が終わったら解放する
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()