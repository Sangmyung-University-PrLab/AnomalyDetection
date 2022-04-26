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
import cv2
import timm
import numpy as np
import pandas as pd

import torch
import albumentations
import albumentations.pytorch
from tqdm import tqdm

class TestDatasetABNORM(torch.utils.data.Dataset):
    def __init__(self, image_dir, dataset_df, transforms):        
        self.image_dir = image_dir
        self.image_df = dataset_df["file_name"].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_df)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_dir +self.image_df[index])
        image = cv2.resize(image, dsize=(512,512))
        # label = self.train_labels[index]
        label = -1

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0
        return image, label
    

def test():
    train_set = pd.read_csv("./open/train_df_aug.csv")
    data_set = pd.read_csv("./open/test_df.csv")
    image_dir = "open/test/"
    
    train_labels = train_set["label"]
    label_unique = sorted(np.unique(train_labels))
    label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
    
    transforms = {
        'test' : albumentations.Compose([        
            albumentations.pytorch.ToTensorV2(),
            ]),
    }
    # train, test split
    
    val_dataset = TestDatasetABNORM(image_dir=image_dir, dataset_df=data_set, transforms=transforms['test'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=8)
    model = timm.create_model('efficientnet_b0', num_classes=88)
    device = torch.device('cpu')
    ckpt_path = "./version4/abnoraml_epoch=58_valid_acc=1.00_valid_loss=0.47.ckpt"
    csv_path = ckpt_path.replace(".ckpt", ".csv")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.to(device)
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '')] = state_dict.pop(key)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    f_pred = []

    with torch.no_grad():
        for batch_idx, samples in (enumerate(tqdm(val_dataloader))):
            x, y = samples
            pred = model(x)
            pred = pred.argmax(1).detach().cpu().numpy().tolist()
            f_pred = f_pred + pred
            
    label_decoder = {val:key for key, val in label_unique.items()}
    f_result = [label_decoder[result] for result in f_pred]
    
    submission = pd.read_csv("./sample_submission.csv")
    submission["label"] = f_result
    submission.to_csv(csv_path, index = False)
    
if __name__ == "__main__":
    test()