import scipy.misc as img
import numpy as np
import sys

if __name__ == '__main__':
	output_image_path = sys.argv[1]
	img_count = 0
	avg_image = None
	for img_path in sys.argv[2:]:
		image = img.imread(img_path)
		if avg_image == None:
			avg_image = np.zeros(image.shape)
		np.add(avg_image, image, avg_image)
		img_count += 1
	np.divide(avg_image, img_count, avg_image)
	img.imsave(output_image_path, avg_image)