# use the code in basics to visualize embedding for resnet on a custom dataset
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import os
from sklearn.externals import joblib
from skimage import io
from PIL import Image
from tensorboardX import SummaryWriter

# You should build custom dataset as below.
class ImageDataset(data.Dataset):
    def __init__(self):
        # 1. Initialize file path or list of file names.
        self.path = '../images'
        self.data = os.listdir(self.path)
        self.transform = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.ToTensor(),
            # for using pretrained models
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # the data has no label, we don't need to train them
        # we just want to see the embedding for this task

        # x = io.imread(os.path.join(self.path, self.data[index]))
        x = Image.open(os.path.join(self.path, self.data[index]))
        x = self.transform(x)
        # the label is random
        return x, 1
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data)


batch_size = 50
data = ImageDataset()
train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=2)

# use resnet pretrained model
resnet = torchvision.models.resnet18(pretrained=True)

extractor = nn.Sequential(*list(resnet.children())[:-1])

for x, y in train_loader:
    embedding = extractor(Variable(x))
    embedding = embedding.view(batch_size, -1)
    break

writer = SummaryWriter('runs/visualize_embedding')
writer.add_embedding(embedding.data, label_img=x, tag="fc", global_step=0)
# writer.add_embedding(embedding)

# # If you want to finetune only top layer of the model.
# for param in resnet.parameters():
#     param.requires_grad = False

# # Replace top layer for finetuning.
# resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is for example.


