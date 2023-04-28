import torch_model as models
import torch
import numpy as np
import cv2
from data import RandomAudioDataset, RandomDotsDataset, AudioVisualDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import io
import utils
import torch.nn.functional as F


def log_softmax(x):
    c = x.max()
    logsumexp = np.exp(x - c).sum()
    return x - c - logsumexp


return_type = "c_concat"
# val_data = RandomDotsDataset(split="val")
val_data = AudioVisualDataset(
    "./data/outputs/val", split="test", data_return_type=return_type, congruous=True
)
# val_data = RandomDotsDataset(split="test")
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

out_size = val_data.num_labels
in_channels = val_data.in_channels
model = models.AudioVisualModel(
    in_channels, out_size
)  # models.MultiNet(in_channels, out_size)
checkpoint = torch.load(
    "checkpoints/091b6c3bf0e742bea0f181ad6d34549a_CE_audio_visual.ckp"
)  # BCE congruous

# checkpoint = torch.load(
#     "checkpoints/e38330ad8c904f75ae731b9e15c158c2_CE_audio_visual.ckp"
# )  # BCE non congruous
model.load_state_dict(checkpoint["state_dict"])

model = model.eval()
model = model.cuda()

mean_positions = val_data.positions  # val_data.xmean_positions  #
outpath = "outputs/output_audio_visual_BCE.avi"
# outpath = "outputs/audio_new_CE.avi"
fps = 20.0
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(outpath, fourcc, fps, (1280, 480))

i = 0
for input, label in tqdm(val_dataloader):

    input = input.cuda()
    model_pred = model(input)
    # ypred = F.log_softmax(model_pred).detach().cpu().numpy()[0]
    ypred = F.softmax(model_pred, dim=1).detach().cpu().numpy()[0]
    # ypred = model_pred.detach().cpu().numpy()[0]
    # print(ypred)
    # ypred = log_softmax(ypred) #F.logsigmoid(ypred).cpu().numpy()[0] #l
    # print(ypred)
    # ypred = torch.log_softmax(ypred)
    # ytrue = np.zeros(out_size)
    # idx = label.numpy()[0].argmax()
    # ytrue[idx] = 1.0
    ytrue = label.detach().cpu().numpy()[0]
    # print(ytrue)
    plt.cla()

    plt = utils.plot_barplot(mean_positions, ypred, "Predicted", "x.jpg")
    ypred_img = utils.read_image_grayscale("x.jpg")  # .astype(np.uint8)
    # ypred_arr = cv2.resize(ypred_img, (250, 250))

    plt.cla()
    plt = utils.plot_barplot(mean_positions, ytrue, "True", "y.jpg")
    ytrue_img = utils.read_image_grayscale("y.jpg")  # .astype(np.uint8)
    # ytrue_arr = cv2.resize(ytrue_img, (250, 250))

    frame = np.zeros(
        (500, 500), dtype=np.uint8
    )  # input.detach().cpu().numpy()[0].reshape(500, 500).astype(np.uint8)
    # dot_indices = np.argwhere(frame==1)
    # if len(dot_indices) > 0:
    #     frame = utils.draw_dots(frame, dot_indices)

    # frame = utils.draw_text(frame, str(idx))
    frame = utils.numpy_to_gray(frame).astype(np.uint8)
    frame = frame * 255
    i += 1
    barplots = cv2.hconcat([ytrue_img, ypred_img])
    # final_frame = cv2.vconcat([barplots, frame])
    # print(final_frame.shape)
    video.write(barplots)
    cv2.imwrite("frame.jpg", barplots)
# print("Accuracy {}".format(np.sum(ytrue == ypred)/len(ytrue)))
video.release()
