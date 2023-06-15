import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import BioactivityDataset, Collater
from sklearn.metrics import confusion_matrix, f1_score


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.6):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.resweight = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, x):
        return x + self.block(x) * self.resweight


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout)
        )
        self.resweight = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, x):
        return self.conv1(x) + self.block(x) * self.resweight


class System(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_gpus = torch.cuda.device_count()
        self.train_bsz = args.batch_size
        self.val_bsz = args.batch_size
        self.num_workers = args.num_workers

        self.placeholder = 6

        ### Percentage of each class in the dataset - multiple softmax classifications
        label_prct = torch.FloatTensor([0.4320725001, 0.3171132153, 0.1782923951, 0.04641016192, 0.01636920724, 0.009742520239])
        class_weights = torch.FloatTensor(1-label_prct)

        ### Model
        self.model = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(6144, 256),
            nn.Sequential(*(ResBlock(256, 256) for _ in range(5))),
            nn.Linear(256, 354),
        )

        ### For multiple softmax classifications
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, reduction = 'none')


    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), 6, 59)


    def training_step(self, batch, batch_idx, training=False):
        x, y = batch

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.LongTensor)

        y_pred = self.forward(x)
        y_pred = y_pred.type(torch.cuda.FloatTensor)

        y = torch.squeeze(y)

        mask = torch.where(y == self.placeholder, torch.zeros(1, device=self.device), torch.ones(1, device=self.device))
        y_target = torch.where(y == self.placeholder, torch.tensor(self.placeholder-1).type_as(y), y)

        loss = self.ce_loss(y_pred, y_target)
        loss = torch.mul(mask, loss)
        loss = loss.mean()

        with torch.no_grad():
            preds = y_pred.argmax(dim=-2)
            new_preds = torch.mul(mask, preds)
            new_y = torch.mul(mask, y)
            total = sum(sum(mask))

            num_no_data = (new_y.shape[0] * new_y.shape[1]) - total
            num_right = sum(sum(new_preds == new_y).float())
            total_right = num_right - num_no_data
            acc = total_right / total

        self.log("train loss", loss, on_epoch=True)
        self.log("train accuracy", acc, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.LongTensor)

        y_pred = self.forward(x)
        y_pred = y_pred.type(torch.cuda.FloatTensor)

        y = torch.squeeze(y)

        mask = torch.where(y == self.placeholder, torch.zeros(1, device=self.device), torch.ones(1, device=self.device))
        y_target = torch.where(y == self.placeholder, torch.tensor(self.placeholder-1).type_as(y), y)

        val_loss = self.ce_loss(y_pred, y_target)
        val_loss = torch.mul(mask, val_loss)
        val_loss = val_loss.mean()

        with torch.no_grad():
            preds = y_pred.argmax(dim=-2)
            new_preds = torch.mul(mask, preds)
            new_y = torch.mul(mask, y)
            total = sum(sum(mask))

            num_no_data = (new_y.shape[0] * new_y.shape[1]) - total
            num_right = sum(sum(new_preds == new_y).float())
            total_right = num_right - num_no_data
            val_acc = total_right / total

        self.log("validation loss", val_loss, on_epoch=True)
        self.log("validation accuracy", val_acc, on_epoch=True)


    def predict_step(self, batch, batch_idx):
        x = batch
        x = x.type(torch.cuda.FloatTensor)

        y_pred = self.forward(x)
        y_pred = y_pred.type(torch.FloatTensor)
        with torch.no_grad():
            preds = y_pred.argmax(dim=-2)

        return preds


    def configure_optimizers(self):
        effect_bsz = self.num_gpus * self.train_bsz * self.args.grad_acc
        scaled_lr = self.args.lr * \
            math.sqrt(effect_bsz) if self.args.lr else None
        print('Effective learning rate:', scaled_lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=scaled_lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: max(
                1 - e / self.args.max_steps, scaled_lr / 1000)
        )
        return [optimizer], [scheduler]


    def train_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + 'train.pkl')
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.train_bsz,
            shuffle=True,
            collate_fn=Collater(),
            pin_memory=True)


    def val_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + 'valid.pkl')
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=Collater(),
            pin_memory=True)


    def predict_dataloader(self):
        dataset = BioactivityDataset(self.args.data_path + 'labtest720.pkl', test=True)
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            pin_memory=True)
