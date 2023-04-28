import os
import shutil
import numpy as np
from tqdm import tqdm


input_dir = "./spectrograms_updated"

train_dir = "./outputs_freqs/train"
val_dir = "./outputs_freqs/val"

positions = os.listdir(input_dir)

for position in tqdm(positions):
    images = os.listdir(os.path.join(input_dir, position, "left"))
    os.makedirs(os.path.join(train_dir, position, "left"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, position, "right"), exist_ok=True)

    os.makedirs(os.path.join(val_dir, position, "left"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, position, "right"), exist_ok=True)

    np.random.shuffle(images)
    val_images = images[:100]
    train_images = images[100:]

    print(len(val_images), len(train_images))

    for train_image in train_images:
        src_path_left = os.path.join(input_dir, position, "left", train_image)
        src_path_right = os.path.join(input_dir, position, "right", train_image)

        dst_path_left = os.path.join(train_dir, position, "left", train_image)
        dst_path_right = os.path.join(train_dir, position, "right", train_image)

        shutil.copyfile(src_path_left, dst_path_left)
        shutil.copyfile(src_path_right, dst_path_right)

    for val_image in val_images:
        src_path_left = os.path.join(input_dir, position, "left", val_image)
        src_path_right = os.path.join(input_dir, position, "right", val_image)

        dst_path_left = os.path.join(val_dir, position, "left", val_image)
        dst_path_right = os.path.join(val_dir, position, "right", val_image)

        shutil.copyfile(src_path_left, dst_path_left)
        shutil.copyfile(src_path_right, dst_path_right)
