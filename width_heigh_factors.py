import os
import numpy as np
import matplotlib.image as mpimg
import sys
import random

if __name__ == '__main__':
	factors = set()
	for img_path in sys.argv[1:]:
		image_name = os.path.basename(img_path)
		image = mpimg.imread(img_path)
		width = float(image.shape[2])
		height = float(image.shape[1])
		print(image.shape, width, height)
		factor = width / height
		str_factor = "{:.2f}".format(factor)
		print(str_factor)
		factors.add(str_factor)
	print("Image ratios: ", factors)