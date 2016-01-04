__author__ = 'alfiya'
from load_tid import load_data
from fft_quality import fft_based_quality
import numpy as np

PATH2IMGS = "/Users/alfiya/Documents/work/Image Quality/tid2013/distorted_images"
PATH2FILE = "/Users/alfiya/Documents/work/Image Quality/tid2013/mos_with_names.txt"

X, y = load_data(PATH2FILE, PATH2IMGS)

quality_scores = np.vectorize(fft_based_quality)(X[:20])

print(quality_scores.shape)
print(np.vstack((X[:20], y[:20], quality_scores)).transpose())
