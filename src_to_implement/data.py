from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision.transforms as tvt

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        transform_list = [
            tvt.ToPILImage()
        ]

        # Data Augmentation: Only apply extra steps if in training mode
        if self.mode == 'train':
            # Inserting augmentation at the beginning of the list (on PIL image)
            transform_list.append(tvt.RandomHorizontalFlip(p=0.5)) # Flipped horizontally
            transform_list.append(tvt.RandomVerticalFlip(p=0.5))   # Flipped vertically

        transform_list.extend([tvt.ToTensor(),
                               tvt.Normalize(mean=train_mean, std=train_std)
                               ])

        self.transform = tvt.Compose(transform_list)

    def __len__(self):
        # Returns the total number of samples in the dataframe
        return len(self.data)

    def __getitem__(self, index):
        """
        Overwrite the method getitem (self,index), which returns the sample as a tuple:
        the image and the corresponding label. Since our raw data is grayscale you need to
        convert the image to rgb using the skimage.color.gray2rgb(*args) function. Before
        returning the sample, perform the transformations specied in the transform member.
        The two return values need to be of type torch.tensor.
        """
        # 1. Get image path and labels from dataframe
        # Assuming column 0 is path, 1 is crack, 2 is inactive
        img_path = self.data.iloc[index, 0]
        labels = self.data.iloc[index, [1, 2]].values.astype('float32')

        # 2. Load image and convert grayscale to RGB
        img = imread(img_path)
        img_rgb = gray2rgb(img)

        # 3. Apply the torchvision transforms
        img_tensor = self.transform(img_rgb)

        # 4. Convert label to torch tensor
        label_tensor = torch.from_numpy(labels)

        return img_tensor, label_tensor
