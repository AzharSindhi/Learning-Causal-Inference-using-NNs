import torch as t
from data import RandomDotsDataset, RandomAudioDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import torch_model as models
from torch.optim import lr_scheduler
import mlflow

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

# hyperparameters

# choose data_type from 'branch' 'left_only' and 'right_only'
# 'left_only' : use only the left ear spectrogram for training
# 'right_only': use only the right ear spectrogram for training
# 'branch': use both left and right ear spectrogram for training using branch model
# 'c_concat': concatenate the left and right spectrogram and use it for training
data_type = "c_concat"
mlflow.set_tracking_uri("http://localhost:5000/")
experiment_name = data_type
experiment = mlflow.set_experiment(experiment_name)
mlflow.start_run(experiment_id=experiment.experiment_id, run_name=data_type)
run = mlflow.active_run()
run_id = run.info.run_id
mlflow.log_param("run_id", run_id)

batch_size = 256
epochs = 15
early_stopping_patience = 3
num_workers = 4
shuffle = True
train_path = "./data/outputs/train"
val_path = "./data/outputs/val"
checkpoint_path = "checkpoints/{}_BCE.ckp".format(run_id)
lr = 0.001
mom = 0.9
fscore_offset_error = 10
# optimizer =

train_dataset = RandomAudioDataset(train_path, data_return_type=data_type)
train_dataloader = t.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)

val_dataset = RandomAudioDataset(val_path, data_return_type=data_type)
val_dataloader = t.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

num_classes = train_dataset.num_labels
in_channels = train_dataset.in_channels

mlflow.log_param("checkpoint_path", checkpoint_path)
mlflow.log_param("learning rate", lr)
mlflow.log_param("batch size", batch_size)
mlflow.log_param("epochs", epochs)
mlflow.log_param("input_shape", train_dataset.input_shape)
mlflow.log_param("in_channels", train_dataset.in_channels)
mlflow.log_param("num_classes", train_dataset.num_labels)
mlflow.log_param(
    "positions", (train_dataset.start_x, train_dataset.end_x, train_dataset.step)
)
mlflow.log_param("position offset", fscore_offset_error)
mlflow.log_param("return type", data_type)
# create an instance of our model
if data_type == "branch":
    model = models.MultiBranch(in_channels, num_classes)
else:
    model = models.MultiNet(in_channels, num_classes)  # .get_resnet18()

mlflow.log_param("model", model.__name__)
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.CrossEntropyLoss()  # t.nn.MSELoss()
mlflow.log_param("loss", str(type(criterion)))
# criterion = custom_l1_loss
# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=lr)
mlflow.log_param("optimizer", str(type(optimizer)))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    model,
    criterion,
    optimizer,
    exp_lr_scheduler,
    train_dataloader,
    val_dataloader,
    cuda=True,
    early_stopping_patience=early_stopping_patience,
    checkpoint_path=checkpoint_path,
    position_offset=fscore_offset_error,
)

# trainer.train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders,
#         dataset_sizes, device = "cuda:0", num_epochs=25)
# # go, go, go... call fit on trainer
res = trainer.fit(epochs)
mlflow.end_run()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label="train loss")
plt.plot(np.arange(len(res[1])), res[1], label="val loss")
plt.yscale("log")
plt.legend()
plt.savefig("losses.png")
