__author__ = 'alfiya'
import numpy as np
import cv2


def level_of_detail(edge_map, threshold=32, block_size=(8, 8)):
    if isinstance(edge_map, str):
        img = cv2.imread(edge_map, 0)
        edge_map = cv2.Laplacian(img, cv2.CV_64F)

    delta = np.zeros(edge_map.shape)
    i_step = block_size[0]
    j_step = block_size[1]
    denom = float(block_size[0]) * block_size[1]
    for i in np.arange(0, edge_map.shape[0], i_step):
        for j in np.arange(0, edge_map.shape[1], j_step):
            delta[i:i + i_step, j:j + j_step] =\
                (edge_map[i:i + i_step, j:j + j_step] > threshold).sum() / denom

    return delta


def noise_type(img):
    if isinstance(img, str):
        img = cv2.imread(img, 0)

    noise = np.array([])

    for i in np.arange(img.shape[0]):
        for j in np.arange(img.shape[1]):
            if ((img[i, j] != img[i + 1, j]) and (img[i, j] != img[i - 1, j])
                 and (img[i, j] != img[i, j + 1]) and (img[i, j] != img[i, j - 1])):
                noise = np.append(noise, img[i, j])

    nn = float(noise.size)
    n1 = (noise == 0).sum()
    n2 = (noise == 255).sum()
    n3 = ((noise > 0) & (noise < 255))
    if n3 / nn < 0.75:
        n_type = "salt and pepper"
    elif n1 / nn < 0.025 or n2 / nn < 0.025:
        n_type = "gaussian"
    else:
        n_type = "random"

    return n_type

