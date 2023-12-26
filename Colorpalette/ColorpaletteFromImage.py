import argparse
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# --------------- Arguments ---------------

parser = argparse.ArgumentParser(description='Colorpalette')
parser.add_argument('--img', type=str, required=True)
args = parser.parse_args()


def get_dominant_colors(infile):
	"Make a color palette"
	
	DOMAIN_COLOR_NUM = 10
	
	image = Image.open(infile)
	w, h = image.size
	print("input_img_size:",w,h)

	if(w > h):
		small_w = 100
		small_h = (h * 100 / w)
	else:
		small_w = (w * 100 / h)
		small_h = 100

	#print(w,h,small_w,small_h)
			
	small_image = image.resize((int(small_w), int(small_h)))
	
	import matplotlib.pyplot as plt
	plt.imshow(small_image)
	plt.show()

	# image with only 10 dominating colors
	result = small_image.convert("P", palette=Image.ADAPTIVE, colors=DOMAIN_COLOR_NUM)
	
	# Find dominant colors
	palette = result.getpalette()
	color_counts = sorted(result.getcolors(), reverse=True)
	colors = list()
	
	for i in range(DOMAIN_COLOR_NUM):
		palette_index = color_counts[i][1]
		dominant_color = palette[palette_index * 3 : palette_index * 3 + 3]
		colors.append(tuple(dominant_color))


	# save colorpalette full image
	os.makedirs("colorpalette", exist_ok=True)
	for i in range(len(colors)):
		color_img = np.zeros((h,w,3), dtype=np.uint8)
		color_img[0:h, 0:w]=colors[i]
		pil_color_img = Image.fromarray(color_img)
		file_name = "colorpalette/" + "color_0" + str(i) + ".jpg"
		pil_color_img.save(file_name)

	print("colorpalette:",colors)
	print("output_colorpalette_img_size:",color_img.shape[1],color_img.shape[0],len(colors))

	return colors

def plot_img_tile(colors):
	"List images in tiles"
	
	imgs = []
	tile_w = 60
	tile_h = 80

	for i in range(len(colors)):
		color_img = np.zeros((tile_h,tile_w,3), dtype=np.uint8)
		color_img[0:tile_h, 0:tile_w]=colors[i]
		imgs.append(color_img)

	fig, ax = plt.subplots(2, int(len(colors)/2), figsize=(5, 5))
	#fig.subplots_adjust(hspace=0, wspace=0)
	for i in range(2):
		for j in range(int(len(colors)/2)):
			ax[i, j].xaxis.set_major_locator(plt.NullLocator())
			ax[i, j].yaxis.set_major_locator(plt.NullLocator())
			ax[i, j].imshow(imgs[1*i+j], cmap="bone")
	#plt.show()
	plt.savefig('colorpalette_tile.png')
	plt.close()
	return

if __name__ == '__main__':
	# Make a color palette
	colors = get_dominant_colors(args.img)
	# List images in tiles
	plot_img_tile(colors)
