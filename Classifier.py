from PIL import Image
from sklearn.metrics import classification_report
from pyprobar import probar
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
import os
import random
import functools
import operator

# The ten classes of the CIFAR10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
The Classifier class provides methods for generating training, validation and testing data, calculating Cross-Entropy
accuracy, as well as training, testing and loading models.
"""
class Classifier:
    # Constructor for Classifier class
    def __init__(self, data, TRAIN_VAL_SPLIT, EPOCHS, BATCH_SIZE, LEARNING_RATE, loss_func, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        self.data = data

        # Seeds
        np.random.seed(101)
        random.seed(101)

        # Train Hyper-parameters
        self.TRAIN_VAL_SPLIT = TRAIN_VAL_SPLIT
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.loss_func = loss_func
        self.optimizer = optimizer

        # Preprocess transforms
        self.transform_data_augmentation = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_normal = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Data loading
        dataloaders = self.get_dataloaders()
        self.trainloader = dataloaders['TRAIN']
        self.valloader = dataloaders['VAL']
        self.testloader = dataloaders['TEST']

    # imshow is used to show images. Is used in report to depict training images with/without data augmentation
    def imshow(self, img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get_dataloaders is used to generate and return training, testng and validation datasets
    def get_dataloaders(self):
        test_dataset = datasets.CIFAR10(root=f'./{self.data}', train=False, download=True,
                                        transform=self.transform_normal)
        if self.data == "data_with_transforms":
            train_dataset = datasets.CIFAR10(root=f'./{self.data}', train=True, download=True,
                                             transform=self.transform_data_augmentation)
            print("Using augmented train data")
        else:
            train_dataset = datasets.CIFAR10(root=f'./{self.data}', train=True, download=True,
                                             transform=self.transform_normal)
            print("Using normal train data")

        train_dataset_indices = list(range(len(train_dataset)))
        np.random.shuffle(train_dataset_indices)
        train_sampler = SubsetRandomSampler(
            train_dataset_indices[int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset))):])
        val_sampler = SubsetRandomSampler(
            train_dataset_indices[:int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset)))])
        return {"TRAIN": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler,
                                    drop_last=True),
                "VAL": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=val_sampler,
                                  drop_last=True),
                "TEST": DataLoader(test_dataset, batch_size=1)}

    # multi_acc returns the accuracy of a distribution of predicted logits against the ground truth distribution
    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(torch.log_softmax(y_pred, dim=1), dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        return torch.round(correct_pred.sum() / len(correct_pred) * 100)

    # train_model trains the model with set of defined training parameters
    def train_model(self, train_params, save_model=False, save_path="ANN_MODELS", save_name=None, epoch_per_save=10):
        model = CNN_Model(train_params)
        model.to(self.device)
        print(model)

        optimizer = self.optimizer(params=model.parameters(), lr=self.LEARNING_RATE)

        if save_model:
            wandb.init(project='nn-cw')
            wandb.run.name = save_name
            wandb.watch(model)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        accuracy_stats = {'train': [], 'val': []}
        loss_stats = {'train': [], 'val': []}

        print("Beginning training")
        for epoch in range(self.EPOCHS):
            model.train()
            train_epoch_loss, train_epoch_acc = 0, 0
            for X_train_batch, y_train_batch in self.trainloader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)
                # print(y_train_pred)

                train_loss = self.loss_func(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
                if save_model and epoch % epoch_per_save == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss, 'acc': train_acc}, f"{save_path}/{save_name}_epoch{epoch}.pth")
            with torch.no_grad():
                model.eval()
                val_epoch_loss, val_epoch_acc = 0, 0
                for X_val_batch, y_val_batch in self.valloader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_pred = model(X_val_batch)

                    val_loss = self.loss_func(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss / len(self.trainloader))
            loss_stats['val'].append(val_epoch_loss / len(self.valloader))
            accuracy_stats['train'].append(train_epoch_acc / len(self.trainloader))
            accuracy_stats['val'].append(val_epoch_acc / len(self.valloader))

            print(
                f"Epoch {(epoch + 1) + 0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f} | "
                f"Train Acc: {accuracy_stats['train'][-1]:.3f} | Val Acc: {accuracy_stats['val'][-1]:.3f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1], 'Val Loss': loss_stats['val'][-1],
                           'Train Acc': accuracy_stats['train'][-1], 'Val Acc': accuracy_stats['val'][-1]})
        print("Finished Training")
        if save_model:
            torch.save({
                'epoch': self.EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_stats['train'][-1], 'acc': accuracy_stats['train'][-1]},
                f"{save_path}/{save_name}_epoch{self.EPOCHS}.pth")

    # load_model loads a saved model and returns it
    def load_model(self, model_path, params):
        checkpoint = torch.load(model_path)
        model = CNN_Model(params)
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    # test_model is used to test a model against the testing data, and prints a classification report
    def test_model(self, model):
        y_pred, y_ground_truth = [], []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.testloader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)

                y_test_pred = model(X_test_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)

                y_pred.append(y_pred_tag.cpu().numpy())
                y_ground_truth.append(y_test_batch.cpu().numpy())
        print(classification_report(y_ground_truth, y_pred, zero_division=0))

"""
The CNN_Model class implements the convolutional neural network model described in our report. The class also provides
methods for forward-propagation.
"""
class CNN_Model(nn.Module):
    # Constructor for CNN_Model class
    def __init__(self, params):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conv_filters = params.get("conv_filters")
        self.conv_layer = nn.ModuleList()
        for i in range(len(conv_filters) - 1):
            self.conv_layer.append(
                nn.Conv2d(in_channels=conv_filters[i], out_channels=conv_filters[i + 1],
                          kernel_size=params.get("kernel_size"), padding=params.get("padding")))
            if params.get("batch_norm") and i % 2 == 0:
                self.conv_layer.append(nn.BatchNorm2d(conv_filters[i + 1]))
            self.conv_layer.append(params.get("activation"))
            if (i + 1) % 2 == 0:
                self.conv_layer.append(params.get("pool"))
        self.conv_layer.append(nn.Flatten())

        self.drop_out = nn.Dropout(params.get("drop_out"))

        # Need to hard code the num of in_channels for fully connected final layer
        self.fc_out = nn.Linear(8192, len(classes))

    # forward is called with each network propagation to calculates the output Tensors from the input Tensors
    def forward(self, inputs):
        x = inputs
        for i in range(len(self.conv_layer)):
            x = self.conv_layer[i](x)
        # Use below to find the num of in_channels for fully connected final layer
        # print(x.shape[-1])
        return self.fc_out(self.drop_out(x))


if __name__ == "__main__":
    data = "data_without_transforms"

    # Standard Hyper-parameters
    train_val_split = 0.4
    epochs = 15
    epochs_per_save = 5
    batch_size = 64
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam

    # Changeable Params
    params = {"learning_rate": 0.001,
              "conv_filters": [3, 32, 64, 128],
              "kernel_size": 5,
              "padding": 0,
              "batch_norm": False,
              "activation": nn.ReLU(inplace=True),
              "pool": nn.MaxPool2d(kernel_size=2, stride=2),
              "drop_out": 0.5}

    ann = Classifier(data, train_val_split, epochs, batch_size, params.get("learning_rate"), loss_func, optimizer)

    save_name = "standard"
    ann.train_model(params, save_model=True, save_path=f"ANN_MODELS/{save_name}",
                    save_name=save_name, epoch_per_save=epochs_per_save)

    # Below loads a model and tests ir. Need to adjust above params dict to the architecture of model being used:
    # model_path = "ANN_MODELS/standard/standard_epoch15.pth"
    # model = ann.load_model(model_path, params)
    # ann.test_model(model)

    # Below generates a grid of images in one batch & shows them along with their labels:
    # dataiter = iter(ann.trainloader)
    # images, labels = dataiter.next()
    # ann.imshow(make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))