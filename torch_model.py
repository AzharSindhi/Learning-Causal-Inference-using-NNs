from tkinter.tix import Tree
from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor


class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 512)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.fc2(out)
        return out


class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 256)
        self.fc3 = nn.Linear(256, out_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.bn1(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class CustomModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fcnn_input_size = 21
        self.fcnn_hidden_size = 1024
        self.fcnn_outsize = 512
        self.fcnn = NeuralNet(
            self.fcnn_input_size, self.fcnn_hidden_size, self.fcnn_outsize
        )
        self.resnet = self.get_fintetuned_resnet18(use_pretrained=True)
        # self.device = "cuda:0"
        self.fc1 = torch.nn.Linear(1024, 512)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, self.num_classes)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img_x, keypoints_x):
        x = self.resnet(img_x)
        if keypoints_x is not None:
            x_ = self.fcnn(keypoints_x)
            x = torch.concat((x, x_), dim=1)
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        # x = self.sigmoid(x)

        return x

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_resnet18(self, use_pretrained):
        model_ft = models.resnet34(pretrained=use_pretrained)
        return model_ft

    def get_fintetuned_resnet18(self, use_pretrained=True):
        model = self.get_resnet18(use_pretrained)
        # take model upto fully connected layer
        # self.set_parameter_requires_grad(model, True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.fcnn_outsize)

        return model


class MultiNet(nn.Module):
    __name__ = "single_branch"

    def __init__(self, in_channels, out_size) -> None:
        super().__init__()
        # self.conv_filter_size = 12
        # self.input_size = input_size
        # self.v1_outsize = (input_size - self.conv_filter_size) + 1
        # self.v1_numkernels = 16

        self.out_size = out_size
        self.num_kernels = 8
        self.kernel_size = 5
        self.V1_conv = nn.Conv2d(
            in_channels, self.num_kernels, kernel_size=self.kernel_size, padding="valid"
        )
        # max pooling to reduce the size
        pooling_size = 32
        self.max_pool = nn.MaxPool2d(kernel_size=pooling_size)
        self.bn1 = nn.BatchNorm2d(self.num_kernels)
        self.flatten = nn.Flatten()
        conv_outsize = (500 - self.kernel_size + 1) // pooling_size
        flatten_size = conv_outsize * conv_outsize * self.num_kernels
        layer_size = flatten_size // 2
        self.MT_dense = nn.Linear(flatten_size, layer_size)
        self.MST = nn.Linear(layer_size, layer_size // 4)

        self.bn2 = nn.BatchNorm1d(layer_size // 4)
        # optical_flow output layer
        self.out = nn.Linear(layer_size // 4, self.out_size)

    def forward(self, x):
        # reshape to 2D
        x = F.relu(self.V1_conv(x))
        if self.training:
            x = F.dropout2d(x)
        x = self.max_pool(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = F.relu(self.MT_dense(x))
        x = F.relu(self.MST(x))
        # if self.training:
        #     x = F.dropout(x)

        x = self.bn2(x)
        x = self.out(x)
        # x = F.relu(x)
        # x = F.logsigmoid(x)
        # x = F.sigmoid(x)
        return x

    def get_resnet18(self, use_pretrained=False):
        model = models.resnet18(pretrained=use_pretrained)
        # take model upto fully connected layer
        # self.set_parameter_requires_grad(model, True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.out_size)
        return model


class MultiBranch(nn.Module):
    __name__ = "Two-Branch"

    def __init__(self, in_channels, out_size) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_size = out_size

        branch_out_dim = 512
        self.branch1 = MultiNet(in_channels, branch_out_dim)
        self.branch2 = MultiNet(in_channels, branch_out_dim)

        self.fc_network = NeuralNet(2 * branch_out_dim, 512, out_size)

    def forward(self, x):
        xleft = x[:, : self.in_channels, :, :]
        xright = x[:, self.in_channels :, :, :]
        x1 = self.branch1(xleft)
        x2 = self.branch2(xright)
        x = torch.concat((x1, x2), dim=1)
        out = self.fc_network(x)
        return out


class AudioVisualModel(nn.Module):
    __name__ = "audio_visual"

    def __init__(
        self,
        in_channels,
        out_size,
        audio_checkpoint_path=None,
        visual_checkpoint_path=None,
        use_pretrained=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_size = out_size
        model_audio = MultiNet(self.in_channels, self.out_size)
        model_visual = MultiNet(1, self.out_size)
        # print(summary(self.model_audio, (self.in_channels, 500, 500)))
        self.fc_input_size = 450
        self.fcn = FCN(self.fc_input_size, 900, out_size)
        if use_pretrained:
            model_audio = self.load_checkpoint(model_audio, audio_checkpoint_path)
            model_visual = self.load_checkpoint(model_visual, visual_checkpoint_path)

        self.model_audio = self.input_to_embedding(model_audio)
        self.model_visual = self.input_to_embedding(model_visual)

    def input_to_embedding(self, model):
        return_nodes = {"MST": "embedding"}
        model2 = create_feature_extractor(model, return_nodes=return_nodes)
        return model2

    def load_checkpoint(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()  # .cuda()
        return model

    def forward(self, x):
        x_audio = x[:, : self.in_channels, :, :]
        x_visual = x[:, self.in_channels :, :, :]
        with torch.no_grad():
            embeddings_audio = self.model_audio(x_audio)["embedding"]
            embeddings_visual = self.model_visual(x_visual)["embedding"]
            embeddings = torch.concat((embeddings_audio, embeddings_visual), dim=1)
            # embeddings = embeddings.cuda()

        out = self.fcn(embeddings)
        return out


class CustomCNN(nn.Module):
    __name__ = "custom_CNN"

    def __init__(self, in_channels, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    audio_checkpoint = "checkpoints/2b6770c3847d4ae8a7efe136308140ff_BCE.ckp"
    visual_checkpoint = "checkpoints/0811a102898e4892963179d9ae5a6a65_CE_visual.ckp"
    model = AudioVisualModel(
        6, 100, audio_checkpoint, visual_checkpoint, use_pretrained=True
    ).cuda()
    # named_layers = dict(model.model_audio.named_modules())

    x = torch.rand(2, 7, 500, 500).cuda()
    # print(summary(model.model_audio, x.shape[1:]))
    model.forward(x)
