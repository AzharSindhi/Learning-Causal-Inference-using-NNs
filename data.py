from turtle import right
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import io
from PIL import Image
import cv2
import os
import glob


class RandomDotsDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        self.resolution = (500, 500)
        self.start_x = 150
        self.end_x = 350
        self.step = 2
        self.xmean_positions = np.arange(self.start_x, self.end_x, self.step)
        self.num_labels = len(self.xmean_positions)
        self.std = [[150, 0], [0, 150]]
        self.num_dots = 5
        self.in_channels = 1
        if split == "train":
            self.ntrials = 2000
            self.num_frames_per_trial = 300
            self.vstimuli, self.labels = self.generate_dot_stimuli_train()
        else:
            self.ntrials = 10
            self.num_frames_per_trial = 30
            self.vstimuli, self.labels = self.generate_dot_stimuli_val()

        print("total stimuli:", len(self.vstimuli))
        print("Total labels:", self.num_labels)
        # print("labels count:", np.unique(self.labels, return_counts=True))

    def generate_dot_stimuli_train(self):

        vstimuli = []
        labels = []
        for i in tqdm(range(self.num_labels)):
            index = i  # np.random.randint(0, len(self.xmean_positions))
            mean_x = self.xmean_positions[index]
            for k in range(self.num_frames_per_trial):
                mean_y = np.random.randint(150, 350)
                mean = [mean_x, mean_y]
                frame = np.zeros(self.resolution)
                xs, ys = np.random.multivariate_normal(mean, self.std, self.num_dots).T
                xs = xs.astype(int)
                ys = ys.astype(int)
                for x, y in zip(xs, ys):
                    frame[y, x] = 1
                #     cv2.circle(frame, (x, y), radius=3, color=(255, 255, 255), thickness=-1)

                # cv2.imshow("stimuli", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # time.sleep(0.1)

                vstimuli.append(frame)
                labels.append(index)

        # cv2.destroyAllWindows()
        return vstimuli, labels

    def generate_dot_stimuli_val(self):

        vstimuli = []
        labels = []
        for i in tqdm(range(self.ntrials)):
            index = np.random.randint(0, len(self.xmean_positions))
            mean_x = self.xmean_positions[index]
            mean_y = 250  # np.random.randint(150, 350)
            mean = [mean_x, mean_y]
            for k in range(self.num_frames_per_trial):
                frame = np.zeros(self.resolution)
                xs, ys = np.random.multivariate_normal(mean, self.std, self.num_dots).T
                xs = xs.astype(int)
                ys = ys.astype(int)
                for x, y in zip(xs, ys):
                    frame[y, x] = 1
                #     cv2.circle(frame, (x, y), radius=3, color=(255, 255, 255), thickness=-1)

                # cv2.imshow("stimuli", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # time.sleep(0.1)

                vstimuli.append(frame)
                labels.append(index)

        # cv2.destroyAllWindows()
        return vstimuli, labels

    def __len__(self):
        # print("Data length: ", len(self.data))
        return len(self.vstimuli)

    def __getitem__(self, index):
        stimuli = np.array(self.vstimuli[index])
        stimuli = stimuli[np.newaxis]
        stimuli = torch.tensor(stimuli).float()
        x_mean_idx = self.labels[index]
        label = np.zeros(self.num_labels)
        label[x_mean_idx] = 1
        label = torch.tensor(label).float()
        return stimuli, label


class RandomAudioDataset:
    def __init__(self, audio_path, split="train", data_return_type="c_concat"):
        # load dataset
        self.audio_path = audio_path
        self.step = 2
        self.start_x = 150
        self.end_x = 350
        self.positions = np.arange(
            self.start_x, self.end_x, self.step
        )  # np.arange(-100, 100, 2)
        self.num_labels = len(self.positions)
        self.input_shape = (500, 500)
        # self.in_channels = 2
        self.split = split
        filtered_positions = self.positions
        self.minf = -80.0
        self.maxf = 1.9073486e-06
        if split == "test":
            filtered_positions = np.arange(self.start_x, self.end_x, self.step)
            np.random.shuffle(filtered_positions)
            self.image_paths = []
            for pos in filtered_positions:
                pos_images = glob.glob(
                    os.path.join(self.audio_path, str(pos), "left", "*.jpg")
                )
                self.image_paths.extend(pos_images[:100])
        else:
            image_paths = glob.glob(os.path.join(self.audio_path, "*", "left", "*.jpg"))
            self.image_paths = sorted(image_paths, key=lambda x: int(x.split("/")[-3]))
            self.image_paths = [
                img
                for img in self.image_paths
                if int(img.split("/")[-3]) in self.positions
            ]

        self.return_type = data_return_type
        if self.return_type == "c_concat":
            self.in_channels = 2
        else:
            self.in_channels = 1

    def __len__(self):
        return len(self.image_paths)

    def get_processed_stimuli(self, path):
        stimuli = cv2.imread(path, 0)
        stimuli = np.resize(stimuli, self.input_shape)
        stimuli = stimuli / 255.0
        stimuli = stimuli[np.newaxis]
        # if self.in_channels == 1:
        #     stimuli = np.mean(stimuli, axis=2)
        #     stimuli = stimuli[np.newaxis]
        # else:
        #     stimuli = np.transpose(stimuli, (2, 0, 1))
        stimuli = torch.tensor(stimuli).float()
        return stimuli

    def __getitem__(self, index):

        left_path = self.image_paths[index]
        class_num = int(left_path.split("/")[-3])
        x_mean_idx = int(np.where(self.positions == class_num)[0])

        label = np.zeros(self.num_labels)

        label[x_mean_idx] = 1
        label = torch.tensor(label).float()

        right_path = left_path.replace("/left/", "/right/")
        stimuli_left = self.get_processed_stimuli(left_path)
        stimuli_right = self.get_processed_stimuli(right_path)

        if self.return_type == "h_concat":
            stimuli = torch.concat((stimuli_left, stimuli_right), dim=1).resize_(
                self.in_channels, *self.input_shape
            )
        elif self.return_type == "c_concat" or self.return_type == "branch":
            stimuli = torch.concat((stimuli_left, stimuli_right), dim=0)

        elif self.return_type == "left_only":
            stimuli = stimuli_left

        elif self.return_type == "right_only":
            stimuli = stimuli_right

        return stimuli, label


class AudioVisualDataset(RandomAudioDataset):
    def __init__(
        self, audio_path, split="train", data_return_type="c_concat", congruous=True
    ):
        super().__init__(audio_path, split=split, data_return_type=data_return_type)
        self.congruous = congruous
        self.std = [[150, 0], [0, 150]]
        self.num_dots = 5

    def get_visual_stimuli(self, x_pos_idx):
        if not self.congruous:
            # x_pos_idx =#np.random.randint(0, self.num_labels)
            # if x_pos_idx > self.num_labels // 2:
            #     x_pos_idx = np.random.randint(0, self.num_labels // 2 - 10)
            # else:
            #     x_pos_idx = np.random.randint(
            #         self.num_labels // 2 - 10, self.input_shape[0] - 10
            #     )
            x_pos_idx = np.random.randint(0, self.num_labels - 1)  # abs(x_pos_idx - 99)

        stimuli = np.zeros(self.input_shape)
        mean_x = self.positions[x_pos_idx]
        mean_y = 250
        mean = [mean_x, mean_y]
        xs, ys = np.random.multivariate_normal(mean, self.std, self.num_dots).T
        xs = xs.astype(int)
        ys = ys.astype(int)
        for x, y in zip(xs, ys):
            stimuli[y, x] = 1

        stimuli = stimuli[np.newaxis]
        stimuli = torch.tensor(stimuli).float()

        return stimuli, x_pos_idx

    def __getitem__(self, index):

        left_path = self.image_paths[index]
        class_num = int(left_path.split("/")[-3])
        x_mean_idx = int(np.where(self.positions == class_num)[0])
        label = np.zeros(self.num_labels)

        label[x_mean_idx] = 1
        right_path = left_path.replace("/left/", "/right/")
        audio_stimuli_left = self.get_processed_stimuli(left_path)
        audio_stimuli_right = self.get_processed_stimuli(right_path)

        if self.return_type == "h_concat":
            audio_stimuli = torch.concat(
                (audio_stimuli_left, audio_stimuli_right), dim=1
            ).resize_(self.in_channels, *self.input_shape)
        elif self.return_type == "c_concat" or self.return_type == "branch":
            audio_stimuli = torch.concat(
                (audio_stimuli_left, audio_stimuli_right), dim=0
            )

        elif self.return_type == "left_only":
            audio_stimuli = audio_stimuli_left

        elif self.return_type == "right_only":
            audio_stimuli = audio_stimuli_right

        visual_stimuli, visual_pos_idx = self.get_visual_stimuli(x_mean_idx)
        label[visual_pos_idx] = 1.0
        label = torch.tensor(label).float()
        stimuli = torch.concat((audio_stimuli, visual_stimuli), dim=0)

        return stimuli, label


if __name__ == "__main__":
    frame_count = 2048
    batch_size = 5
    # path = "./data/hrir_final.mat"
    path = "./data/outputs/train"
    train_dataset = RandomAudioDataset(path, data_return_type="left_only")
    # # # train_dataset.transform = False
    # # # val_dataloader = DataLoader(train_dataset, batch_size=batch_size,
    # # #                                         shuffle = True, num_workers=0)

    # # for i in range(10):
    # #     X, y = train_dataset[i]
    # #     print(X.shape, y.shape)

    # path = "./data/output/train"
    # transform = transforms.Compose([
    #     transforms.Resize([500, 500]),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0, 0, 0],
    #                           std=[255.0, 255.0, 255.0]),
    # ])
    # # train_dataset = ImageFolder(path, transform=transform)
    # train_dataset = train_dataset.get_audio_dataset_dir(path, transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    for X, y in train_dataloader:
        # Xleft = X[:, :6, :, :]
        # Xright = X[:, 6:, :, :]
        # print(Xleft.shape, Xright.shape, y.shape)
        print(X.shape)
        break
