import matplotlib.pyplot as plt
import numpy as np
import cv2


def numpy_to_gray(img):
    im = np.array(img * 255, dtype=np.uint8)
    # threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    return im


def read_image_grayscale(path):
    # im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # im_gray[im_gray < 255] = 0
    im_gray = cv2.imread(path)
    im_gray = cv2.convertScaleAbs(im_gray, alpha=(255.0 / 65535.0)) * 255
    # thresh = 127
    # im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_gray


def rgb_to_gray(image):
    pass


def draw_dots(img, dot_indices):

    for dot_idx in dot_indices:
        cv2.circle(
            img, (dot_idx[1], dot_idx[0]), radius=3, color=(255, 255, 255), thickness=-1
        )

    return img


def draw_text(image, text, org=(240, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(
        image, text, org, font, fontScale, color, thickness, cv2.LINE_AA
    )
    print(type(image))
    return image


def convert_plt_to_numpy(plt):
    pass


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))


def plot_barplot(x, y, label, save_file):

    # x = #range(len(ytrue))
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel("Xmean position")
    low = 0
    high = 1
    plt.ylim([low, high])
    ypred_fig = plt.bar(x, y)
    plt.savefig(save_file)
    return plt
