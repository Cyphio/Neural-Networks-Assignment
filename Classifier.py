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
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
import os
import random

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Classifier:
    def __init__(self, ANN_flag):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # Seeds
        np.random.seed(101)
        random.seed(101)

        # ANN hyper-parameters
        self.TRAIN_VAL_SPLIT = 0.4
        self.EPOCHS = 15
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.0001
        # For SGD
        self.MOMENTUM = 0.9

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam

        if ANN_flag == 'MLP':
            self.INPUT_SIZE = 32 * 32 * 3
            self.model_name = 'MLP'
            self.model_class = MLP_Model(self.INPUT_SIZE)
        if ANN_flag == 'CNN':
            self.INPUT_SIZE = 32 * 32
            self.model_name = 'CNN'
            self.model_class = CNN_Model()

        # Preprocess transforms
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Data loading
        dataloaders = self.get_dataloaders()
        self.trainloader = dataloaders['TRAIN']
        self.valloader = dataloaders['VAL']
        self.testloader = dataloaders['TEST']

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def get_dataloaders(self):
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        train_dataset_indices = list(range(len(train_dataset)))
        np.random.shuffle(train_dataset_indices)
        train_sampler = SubsetRandomSampler(train_dataset_indices[int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset))):])
        val_sampler = SubsetRandomSampler(train_dataset_indices[:int(np.floor(self.TRAIN_VAL_SPLIT * len(train_dataset)))])
        return {"TRAIN": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler, drop_last=True),
                "VAL": DataLoader(train_dataset, batch_size=self.BATCH_SIZE, sampler=val_sampler, drop_last=True),
                "TEST": DataLoader(test_dataset, batch_size=1)}

    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(torch.log_softmax(y_pred, dim=1), dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        return torch.round(correct_pred.sum() / len(correct_pred)*100)

    def train_model(self, save_model=False, save_path="ANN_MODELS", save_name=None, epoch_per_save=10):
        model = self.model_class
        model.to(self.device)

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

            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f} | "
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
                'loss': loss_stats['train'][-1], 'acc': accuracy_stats['train'][-1]}, f"{save_path}/{save_name}_epoch{self.EPOCHS}.pth")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = self.model_class
        model.to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

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

class MLP_Model(nn.Module):
    def __init__(self, INPUT_SIZE):
        nn.Module.__init__(self)

        LAYER_WIDTHS = [INPUT_SIZE, 64, 64, 64, len(classes)]

        self.linear_layers = nn.ModuleList()
        for i in range(len(LAYER_WIDTHS) - 1):
            self.linear_layers.append(nn.Linear(LAYER_WIDTHS[i], LAYER_WIDTHS[i + 1]))
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        for i in range(len(self.linear_layers)):
            x = self.activation(self.linear_layers[i](x))
        return x

class CNN_Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        INPUT_FILTER = 3
        CONV_FILTERS = [INPUT_FILTER, 3, 3, 3, len(classes)]
        kernel_size = (5, 5)

        self.conv_layers = nn.ModuleList()
        for i in range(len(CONV_FILTERS) - 2):
            self.conv_layers.append(nn.Conv2d(CONV_FILTERS[i], CONV_FILTERS[i + 1], kernel_size))
        self.fc_in_size = CONV_FILTERS[-2] * kernel_size[0] * kernel_size[1]
        print(self.fc_in_size)
        self.fc_out = nn.Linear(self.fc_in_size, CONV_FILTERS[-1])

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.conv_layers)-1):
            # x = self.activation(self.conv_layers[i](x))
            x = self.pool(self.activation(self.conv_layers[i](x)))
        return self.activation(self.fc_out(x.view(-1, self.fc_in_size)))

if __name__ == "__main__":
    ANN_flag = "CNN"
    ann = Classifier(ANN_flag)

    save_name = "CNN_filter-size-3"
    ann.train_model(save_model=True, save_path=f"ANN_MODELS/{ANN_flag}/{save_name}", save_name=save_name, epoch_per_save=5)

    # model_path = "ANN_MODELS/MLP/MLP_h-layer-width-64/MLP_h-layer-width-64_epoch15.pth"
    # model = ann.load_model(model_path)
    # ann.test_model(model)