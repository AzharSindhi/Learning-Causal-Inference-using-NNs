import time
import numpy as np
from scipy.io import loadmat

# from PIL import Image as im
# from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
from tqdm import tqdm
import os
from scipy import signal
import multiprocessing as mp

# Refer to HRIR documentation in the CIPIC dataset
hrir_l = loadmat("hrirInt_left.mat")["hrirInt_l"].T
hrir_r = loadmat("hrirInt_right.mat")["hrirInt_r"].T
noise = loadmat("noise.mat")["noise"][0]


frame_X = 500
frame_Y = 500

azimuths = np.arange(-20, 20.5, 0.5)
num_samples = 736
base_dir = "./spectrogramss"


def angle_to_idx(angle):
    idx = int(np.argmin(np.abs(azimuths - angle)))
    return idx


def get_sound(idx):
    # HRTF is just convolution with left/right
    # ear impulse responses at a given angle.
    x = np.random.randn(num_samples)
    x_l = np.convolve(x, hrir_l[idx], mode="same").flatten().astype("float32")
    x_r = np.convolve(x, hrir_r[idx], mode="same").flatten().astype("float32")
    # normalize between -1 and 1
    x_l = x_l / np.max(np.abs(x_l))
    x_r = x_r / np.max(np.abs(x_r))

    return x_l, x_r


def get_angle(x, y):
    x, y = x - 256, 512 - y
    angle = 90 - np.arctan2(y, x) * 180 / np.pi
    dist = 1 - (x**2 + y**2) ** 0.5 / (256**2 + 512**2) ** 0.5
    i = angle_to_idx(angle)

    return i, dist


def save_spectrogram(outpath, y, sr=44100):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=len(y))
    log_ms = librosa.power_to_db(ms, ref=np.max)
    # print(log_ms.shape)
    # np.save(outpath, log_ms)
    # print("saved")
    # return log_ms.max(), log_ms.min()
    librosa.display.specshow(log_ms, sr=sr, cmap="gray")
    # ax.set_title(title)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Frequency")
    # ax.set_xlim(left=0, right=28)
    plt.savefig(outpath, format="jpg")
    plt.close(fig)


def sound_position(x):
    outdir_left = os.path.join(base_dir, str(x), "left")
    outdir_right = os.path.join(base_dir, str(x), "right")
    os.makedirs(outdir_left, exist_ok=True)
    os.makedirs(outdir_right, exist_ok=True)
    idx = angle_to_idx(angles[x])
    for k in range(400):
        left_conv, right_conv = get_sound(idx)
        outpath_left = os.path.join(outdir_left, str(k) + ".jpg")
        outpath_right = os.path.join(outdir_right, str(k) + ".jpg")
        save_spectrogram(outpath_left, left_conv)
        save_spectrogram(outpath_right, right_conv)


if __name__ == "__main__":
    y = 256  # constant

    xpositions = np.arange(0, frame_X, 2)
    angles = np.linspace(-20, 20, frame_X, endpoint=True)

    with mp.Pool(mp.cpu_count()) as p:
        list(tqdm(p.imap(sound_position, xpositions), total=len(xpositions)))
