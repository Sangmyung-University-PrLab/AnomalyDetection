# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import random
import cv2
import timm
import numpy as np
import mlflow
import pandas as pd

import torch
import torchvision.transforms as T
import albumentations
import albumentations.pytorch
from torch.nn import functional as F
from torchmetrics import Accuracy

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({"early_stopping.monitor": "valid_acc", "early_stopping.patience": 10})
        parser.add_lightning_class_args(ModelCheckpoint, "ModelCheckpoint")
        parser.set_defaults({"ModelCheckpoint.monitor": "valid_loss", "ModelCheckpoint.filename":"abnomaly_{epoch:02d}_{valid_acc:.2f}_{valid_loss:.2f}",\
            "ModelCheckpoint.save_top_k": 5})
        parser.set_defaults({"trainer.max_epochs": 300})
        parser.set_defaults({"trainer.min_epochs": 100})


class DatasetABNORM(torch.utils.data.Dataset):
    def __init__(self, image_dir, dataset_df, transforms):        
        self.image_dir = image_dir
        self.image_df = dataset_df["file_name"].tolist()
        self.labels = dataset_df["label"].tolist()
        label_unique = sorted(np.unique(self.labels))
        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        self.train_labels = [label_unique[k] for k in self.labels]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.train_labels)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_dir +self.image_df[index])
        image = cv2.resize(image, dsize=(512,512))
        label = self.train_labels[index]

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0
        return image, label

        
class ImageClassifier(LightningModule):
    def __init__(self, model, lr=0.001, gamma=0.7, smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model
        self.model = timm.create_model(model, pretrained=True, num_classes=88)
        self.val_acc = Accuracy()
        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=self.hparams.smoothing)
        for key, val in self.hparams.items():
            mlflow.log_param(key, val)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        self.log("smoothing", self.hparams.smoothing)
        self.log("train_loss", loss)
        self.log("train_lr", self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        self.val_acc(pred, y.int())
        self.log("valid_acc", self.val_acc.compute())
        self.log("valid_loss", loss)
        self.val_acc.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=self.hparams.gamma)

        return (
            {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "valid_loss",
                },
            }
        )


class ABNORMdataModule(LightningDataModule):
    def __init__(self, batch_size=8, csv_fn="open/train_df_aug.csv", image_dir = "./open/train/"):
        super().__init__()
        self.save_hyperparameters()
        self.data_set = pd.read_csv(csv_fn)
        self.image_dir = image_dir
        # data shuffle
        self.data_set = self.data_set.sample(frac=1).reset_index(drop=True)
        
        # train, test split
        self.train_data = self.data_set.sample(frac=0.85,random_state=42) #random state is a seed value
        self.valid_data = self.data_set.drop(self.train_data.index)

        self.transforms = {
            'train' : albumentations.Compose([
                    albumentations.RandomRotate90(),
                    albumentations.GaussNoise(),
                    albumentations.ColorJitter(),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.RandomBrightnessContrast(p=0.33),
                    albumentations.OneOf([
                        albumentations.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                        albumentations.ShiftScaleRotate(rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, p=1),        
                        albumentations.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
                    ], p=1),    
                    albumentations.CoarseDropout(max_holes=16, max_height=50, max_width=50, fill_value=0),
                    albumentations.pytorch.ToTensorV2(),
                ]),
            'valid' : albumentations.Compose([        
                albumentations.pytorch.ToTensorV2(),
                ]),
            'test' : albumentations.Compose([        
                albumentations.pytorch.ToTensorV2(),
                ]),
        }

    def train_dataloader(self):
        train_dataset = DatasetABNORM(
                image_dir=self.image_dir,
                dataset_df=self.train_data,
                transforms=self.transforms['train']
            )
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=8)

    def val_dataloader(self):
        val_dataset = DatasetABNORM(
                image_dir=self.image_dir,
                dataset_df=self.valid_data,
                transforms=self.transforms['valid']
            )
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size, num_workers=8)


def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = CustomLightningCLI(
        ImageClassifier, ABNORMdataModule, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://prserver.iptime.org:9650')  # set up connection
    mlflow.set_experiment('')  # set the experiment
    mlflow.start_run() #run mlflow logger
    cli_main()
    mlflow.end_run()