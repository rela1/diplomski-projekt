import os
import numpy as np
import matplotlib.image as mpimg
import sys
import random
from skimage import exposure

if __name__ == '__main__':
	X = []
	y = []
	output_folder = sys.argv[1]
	name_prefix = sys.argv[2]
	histogram_equalize = sys.argv[3]
	num_images = len(sys.argv) - 4
	count = 0
	for img_path in sys.argv[4:]:
		image_name = os.path.basename(img_path)
		image = mpimg.imread(img_path)
		if histogram_equalize == "True":
			image = exposure.equalize_adapthist(image, clip_limit=0.03)
			assert image.shape[-1] == 3
		img_path_parts = image_name.split("_")
		label = int(img_path_parts[0])
		index = int(img_path_parts[1].split(".")[0])
		X.append(image)
		if label == 1:
			y.append(1)
		else:
			y.append(0)
		if count  % 10 == 0:
			print("Done with {}/{} images...".format(count, num_images))
		count += 1
	print("Done with {}/{} images...".format(count, num_images))
	X = np.array(X)
	y = np.array(y)
	np.save(os.path.join(output_folder, name_prefix + "_X"), X)
	np.save(os.path.join(output_folder, name_prefix + "_y"), y)
	print(name_prefix + "_X shape {}, ", name_prefix + "_y shape {}".format(X.shape, y.shape))
	print(name_prefix + " dataset statistics:")
	print("\tpositive examples: ", np.sum(y == 1))
	print("\tnegative examples: ", np.sum(y == 0))