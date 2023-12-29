#!/bin/bash

# 元の画像が格納されているフォルダのパス
source_folder="/stable-diffusion-webui/outputs/moasic"

# コピー先のフォルダのパス
destination_folder="/stable-diffusion-webui/outputs/auto_moasic"

# ランダムに10枚のPNG画像を選択
shuf -n 10 -e "$source_folder"/*.png | while read -r image_path; do
  # 画像をコピー
  cp "$image_path" "$destination_folder/"
done

echo "some images copied from $source_folder to $destination_folder"
