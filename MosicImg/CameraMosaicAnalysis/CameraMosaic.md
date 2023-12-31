# camera_mosaic

## Overview

max configured size is 200 MPixelsで、一つのタイル画像は512x512なので、
一度のスクリプトで、200,000,000 ÷ 262,144 ≈ 763.24枚である。
63x63モザイクに必要な約5000枚の画像を集めるためには、約7回のスクリプト試行が必要である。
samplingstepは10として生成画像の精度と計算時間のバランスを探った。

## Usage

1回目の試行では、
cinema camera, close up shot, Key Lighting, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple
0-10000[100]
として、700枚の画像を生成した。狙いはベースとなるカラーバリエーションの増強である。
生成時間は10分程度であった。

2回目の試行では、
ARRI cinema camera, Key Lighting, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple
10000-20000[100]
として、700枚の画像を生成した。狙いはベースとなるカラーバリエーションの変化である。
生成時間は10分程度であった。
close up shotを外すことで、彩度を低下させる効果が得られた。

3回目の試行では、
ARRI cinema camera, Soft color, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
30000-40000[77]
として、700枚の画像を生成した。狙いは彩度が全体的に高いので、落とすためである。
生成時間は10分程度であった。
Soft colorを外すことで、彩度を安定させる効果が得られた。

4回目の試行では、
ARRI cinema camera, Soft color, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
10000-20000[77]
として、700枚の画像を生成した。狙いは彩度が全体的に高いので、落とすためである。
生成時間は10分程度であった。
Soft colorを外すことで、彩度を安定させる効果が得られた。

5回目の試行では、
ARRI cinema camera, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
10000-20000[15]
AnythingV5lnk,aniflatmixAnimeFlatColorStyle,breakdomain,cetuMix,flat2Danimerge
として、700枚の画像を生成した。狙いはベースを保ったまま、彩度と絵柄のバリエーションを増やすことである。
生成時間は10分程度であった。

6回目の試行では、
ARRI cinema camera, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
40000-50000[25]
AnythingV5lnk,breakdomain,flat2Danimerge
として、700枚の画像を生成した。狙いはベースを保ったまま、彩度と絵柄のバリエーションを増やすことである。
生成時間は10分程度であった。

7回目の試行では、
ARRI cinema camera, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
30000-40000[25]
AnythingV5lnk,breakdomain,flat2Danimerge
として、700枚の画像を生成した。狙いはベースを保ったまま、彩度と絵柄のバリエーションを増やすことである。
生成時間は10分程度であった。

8回目の試行では、
ARRI cinema camera, Dimly Lighting, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
40000-50000[25]
AnythingV5lnk,breakdomain,flat2Danimerge
として、700枚の画像を生成した。狙いはベースを保ったまま、暗部と絵柄のバリエーションを増やすことである。
生成時間は10分程度であった。

9回目の試行では、
ARRI cinema camera, Dimly Lighting, white
worst quality, low quality, nsfw, interlocked fingers, skin blemishes
white, red,  green,  blue, yellow, metal, purple,orange,pink
60000-70000[10]
AnythingV5lnk,breakdomain,flat2Danimerge
として、700枚の画像を生成した。狙いはベースを保ったまま、ファイル名が重なったので補填である。
生成時間は10分程度であった。

## Example

02657-17631