import os
import matplotlib.image as mpimg
from skimage import exposure
import sys

if __name__ == '__main__':
	output_folder = sys.argv[1]
	for img_path in sys.argv[2:]:
		image_name = os.path.basename(img_path)
		image = mpimg.imread(img_path)
		image_normalized = exposure.equalize_adapthist(image, clip_limit=0.03)
		assert image_normalized.shape == image.shape
		assert image_normalized.shape[-1] == 3
		mpimg.imsave(os.path.join(output_folder, image_name), image_normalized)