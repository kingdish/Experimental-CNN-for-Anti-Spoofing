from torch.utils import data
import cv2
import os
import torch
import sys
import pandas as pd
import numpy as np
from PIL import Image
sys.path.append('./utils')
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))


class CASIASURFDataset(data.dataset.Dataset):
    def __init__(self, split="train", img_size=112, transform=None):
        super().__init__()

        self.split = split
        self.root = os.path.join(UTILS_DIR, 'dataset')
        self.df = pd.read_csv(os.path.join(self.root, self.split + "_list.txt"), sep=" ", header=None, names=["color", "depth", "ir","is_real"])
        self.df = self.df.drop(['depth'], axis=1)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        color_img = cv2.imread(os.path.join(self.root, self.df.iloc[idx]["color"]))
        ir_img = cv2.imread(os.path.join(self.root, self.df.iloc[idx]["ir"]), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.resize(color_img, (self.img_size, self.img_size))
        ir_img = cv2.resize(ir_img, (self.img_size, self.img_size))
        img = cv2.merge([color_img, ir_img])
        # merged = torch.from_numpy(merged)
        img = Image.fromarray(img)  # TODO: use either cv2 or PIL, not both
        if self.transform:
            img = self.transform(img)
        # img = (img / 255).astype('float32').reshape([self.img_size, self.img_size, -1])
        # img = torch.tensor(img.transpose(2, 0, 1))

        label = np.array(self.df.iloc[idx]["is_real"])
        label = torch.from_numpy(label)

        return img, label


if __name__ == "__main__":
    # test
    test_data = CASIASURFDataset(split="train")
    #cv2.imread(os.path.join(test_data.root, test_data.df.iloc[idx]["color"]))
    test_img = np.asarray(test_data[0][0])
    print(test_img[:,:,3].shape)
    # print(test_data.df.groupby("is_real").count())
    # print(len(test_data))
    # print(test_data[0])