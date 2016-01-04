__author__ = 'alfiya'
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


def magnitude(imgfile, verbose=False):
    img = cv2.imread(imgfile, 0)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    log_magnitude = np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # scale between 0 and 255
    magnitude_spectrum = 20 * log_magnitude # (log_magnitude - log_magnitude.min()) * 255 / (log_magnitude.max() - log_magnitude.min())
    actual_magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    if verbose:
        print("Image shape", img.shape, "Fourier shape", dft.shape)
        print('Image mean', img.mean(), "Image max", img.max())
        print('Comp 1 min, mean, max', dft[:, :, 0].mean(), dft[:, :, 0].min(), dft[:, :, 0].max())
        print('Comp 2 min, mean, max', dft[:, :, 1].mean(), dft[:, :, 1].min(), dft[:, :, 1].max())
        print('Magnitude mean, min, max', magnitude_spectrum.mean(), magnitude_spectrum.min(), magnitude_spectrum.max())
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(log_magnitude, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
    return magnitude_spectrum, log_magnitude


def circle_mask(X):
    m, n = X.shape
    a, b = int(m / 2), int(n / 2)
    y, x = np.ogrid[-a: m - a, -b: n - b]

    for r in np.arange(max(a, b) + 1, 1, -1):  #  #np.arange(euclidean((a, b), (m, n)), 1, -1)
        mask = x * x + y * y <= r * r

        yield mask


def magnitude_stats(magnitude_spectrum, actual_magnitude, verbose=False):
    total_energy = actual_magnitude.sum()
    energy = np.array([])
    n = 0
    for mask in circle_mask(actual_magnitude):
        circle_energy = actual_magnitude[mask].sum()
        if n == 0:
            total_energy = circle_energy
        energy = np.append(energy, circle_energy / total_energy)
        n += 1

    ideal_energy = 1 - np.arange(n) / float(n)
    mean, mx = magnitude_spectrum.mean(), magnitude_spectrum.max()
    if verbose:
        pass
        # print(mean, mx)
        plt.scatter(np.arange(1, energy.size + 1), energy)
        plt.plot(ideal_energy)
        plt.show()

    return energy, ideal_energy, mean, mx


def noise_score(actual, ideal, mean, mx, E_t=0.035):
    pos = actual > ideal
    E_pos = (actual[pos] - ideal[pos]).sum() / ideal.sum() if pos.any() else 0
    if E_pos > E_t:
        eta = mean / mx
    else:
        eta = 0
    return E_pos, eta


def fft_based_quality(imgfile, verbose=False):
    magnitude_spectrum, actual_magnitude = magnitude(imgfile, verbose)
    ratio, ideal_ratio, mean, mx = magnitude_stats(magnitude_spectrum, actual_magnitude, verbose)
    E_pos, sc = noise_score(ratio, ideal_ratio, mean, mx)
    if verbose:
        print("positive energy", E_pos, "noise level", sc)
    return sc

# def img_quality()
if __name__ == "__main__":
    imgfile = "/Users/alfiya/Documents/work/Image Quality/tid2013/reference_images/goldhill.png"
    print(fft_based_quality(imgfile, verbose=True))

    imgfile = "/Users/alfiya/Documents/work/Image Quality/tid2013/reference_images/I15.bmp"
    print(fft_based_quality(imgfile, verbose=True))

    imgfile = "/Users/alfiya/Documents/work/Image Quality/tid2013/distorted_images/i15_01_5.bmp"
    print(fft_based_quality(imgfile, verbose=True))
    #
    # imgfile = "/Users/alfiya/Documents/work/Image Quality/tid2013/distorted_images/I01_01_1.bmp"
    # print(fft_based_quality(imgfile, verbose=True))