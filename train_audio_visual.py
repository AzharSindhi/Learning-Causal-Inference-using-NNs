import torch as t
from data import AudioVisualDataset
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

# mlflow.set_tracking_uri(uri="/outputs")

# mlflow.set_tracking_uri("http://localhost:4122/")
experiment_name = "audio_visual_model_rawfreqs"  # mlflow.create_experiment("model")
experiment = mlflow.set_experiment(experiment_name)
mlflow.start_run(experiment_id=experiment.experiment_id)
run = mlflow.active_run()
run_id = run.info.run_id
mlflow.log_param("run_id", run_id)

batch_size = 256
epochs = 20
early_stopping_patience = 3
num_workers = 0
shuffle = True
train_path = "./data/outputs/train"
val_path = "./data/outputs/val"
checkpoint_path = "checkpoints/{}_CE_audio_visual.ckp".format(run_id)
audio_checkpoint = "checkpoints/ff96fc86c3fe40d580b8a58c754d0ce2_BCE.ckp"
visual_checkpoint = "checkpoints/0811a102898e4892963179d9ae5a6a65_CE_visual.ckp"
use_pretrained = True
lr = 0.001
mom = 0.9
fscore_offset_error = 10
congruous = True
# optimizer =

train_dataset = AudioVisualDataset(
    train_path, split="train", data_return_type="c_concat", congruous=congruous
)
train_dataloader = t.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)

val_dataset = AudioVisualDataset(
    val_path, split="val", data_return_type="c_concat", congruous=congruous
)
val_dataloader = t.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

num_classes = train_dataset.num_labels
in_channels = train_dataset.in_channels

mlflow.log_param("audio_model_checkpoint", audio_checkpoint)
mlflow.log_param("visual_model_checkpoint", visual_checkpoint)
mlflow.log_param("use_pretrained", use_pretrained)
mlflow.log_param("congruous", congruous)
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
# mlflow.log_param("return type", data_type)
# create an instance of our model

model = models.AudioVisualModel(
    in_channels, num_classes, audio_checkpoint, visual_checkpoint, use_pretrained
)

mlflow.log_param("model", model.__name__)
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCEWithLogitsLoss()  # t.nn.CrossEntropyLoss()  # t.nn.MSELoss()
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
