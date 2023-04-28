import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import os
import copy
import mlflow

cudnn.benchmark = True
# from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,  # Model to be trained.
        crit,  # Loss function
        optim=None,  # Optimizer
        scheduler=None,  # scheduler
        train_dl=None,  # Training data set
        val_test_dl=None,  # Validation (or test) data set
        class_weights=None,  # class weights
        cuda=True,  # Whether to use the GPU
        early_stopping_patience=-1,  # The patience for early stopping
        checkpoint_path=None,
        position_offset=5,
    ):
        self._model = model
        self._crit = crit
        self._optim = optim
        self.scheduler = scheduler
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self.class_weights = class_weights
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.checkpoint_path = checkpoint_path
        self.position_offset = position_offset

        if cuda:
            self._model = model.cuda()
            self._crit = crit  # .cuda()

    def save_checkpoint(self, epoch):
        t.save(
            {"state_dict": self._model.state_dict()},
            self.checkpoint_path,
        )

    def calculate_f1_custom(self, ytrue, ypred):
        ytrue = np.asarray(ytrue)
        ypred = np.asarray(ypred)
        ypred_new = np.abs(ytrue - ypred)
        indices = np.argwhere(ypred_new < self.position_offset)
        ypred_new[indices] = ytrue[indices]

        fscore = f1_score(ytrue, ypred_new, average="macro")
        return fscore

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(
            self.checkpoint_path,
            "cuda" if self._cuda else None,
        )
        self._model.load_state_dict(ckp["state_dict"])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(
            m,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            fn,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable lenght axes
                "output": {0: "batch_size"},
            },
        )

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        out = self._model(x)
        # -calculate the loss
        loss = self._crit(out, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # calculate training f1score
        # ytrue = y.cpu().numpy().argmax().flatten()
        # ypred = out.cpu().numpy().argmax().flatten()
        return loss.item(), out

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model(x)
        # return the loss and the predictions
        loss = self._crit(pred, y)
        return loss.item(), pred

    def train_epoch(self):
        # set training mode
        self._model.train()
        average_loss = 0
        ytrue = []
        ypred = []
        # iterate through the training set
        for data in tqdm(self._train_dl):
            x, y = data
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss, pred = self.train_step(x, y)
            pred = torch.sigmoid(pred)
            yt = y.detach().cpu().numpy().argmax(axis=1).flatten()
            yp = pred.detach().cpu().numpy().argmax(axis=1).flatten()
            ytrue.extend(yt)
            ypred.extend(yp)
            # calculate the average loss for the epoch and return it
            average_loss += loss

        self.scheduler.step()
        fscore = self.calculate_f1_custom(ytrue, ypred)
        average_loss = average_loss / len(self._train_dl)
        print("TRAIN: average loss {:.3f} and f1_score ".format(average_loss), fscore)
        return average_loss, fscore

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        average_test_loss = 0.0
        ypred = []
        ytrue = []
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for data in tqdm(self._val_test_dl):
                x, y = data
                _yt = y.detach().numpy().astype(int)[0]
                _yt = np.argmax(_yt)
                ytrue.append(_yt)
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                pred = torch.sigmoid(pred)
                pred = pred.detach().cpu().numpy()[0]
                pred = np.argmax(pred)
                # print(y.shape, pred.shape)
                ypred.append(pred)
                average_test_loss += loss
                # save the predictions and the labels for each batch

            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions

            fscore = self.calculate_f1_custom(ytrue, ypred)

            average_test_loss = average_test_loss / len(self._val_test_dl)
            print(
                "TEST: average loss and f1 score: {:.3f}, ".format(average_test_loss),
                fscore,
            )

            # return the loss and print the calculated metrics
            return average_test_loss, fscore

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses, val_losses = [], []
        epoch = 0
        same_loss_count = 0
        prev_loss = np.inf
        temp_checkpoint = 1
        for epoch in range(epochs):
            print("EPOCH {} / {}".format(epoch, epochs))
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss, fscore_train = self.train_epoch()
            val_loss, fscore_val = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if train_loss < prev_loss:
                self.save_checkpoint(temp_checkpoint)

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if same_loss_count >= self._early_stopping_patience:
                return train_losses, val_losses

            if train_loss >= prev_loss:
                same_loss_count += 1
            else:
                same_loss_count = 0

            prev_loss = train_loss

            # Log the train and validation scores
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("mean_fscore_train", fscore_train, step=epoch)
            mlflow.log_metric("mean_fscore_val", fscore_val, step=epoch)

        return train_losses, val_losses

    def train_model(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        device="cuda:0",
        num_epochs=25,
    ):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # def save_onnx(self, path):
        #     pass
