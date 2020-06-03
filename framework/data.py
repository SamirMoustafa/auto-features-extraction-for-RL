import os
import random
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import torchvision
from torchvision import transforms
import cv2
from glob import glob
import PIL
from skimage.transform import resize

# make experiments reproducible
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class FeatureExtractor():
    def __init__(self, model, device, data_iterator, checkpoint_path):
        self.model = model
        self.device = device
        self.data_loader = data_iterator
        self.checkpoint_path = checkpoint_path

    def infer(self, loader):
        feature_vec = []
        self.model.load_state_dict(torch.load(self.checkpoint_path)['state_dict'])
        self.model.eval()
        for x_batch in loader:
            x_batch = x_batch.to(self.device)
            features, _ = self.model(x_batch)
            feature_vec.extend(features.cpu().detach().numpy())
        feature_vector = np.array(feature_vec)
        return feature_vector

    def get_features(self):
        X_features = self.infer(self.data_loader)
        return X_features

class ImageTransformation():
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, image):
    first = self.transform(image)
    second = self.transform(image)
    return first, second

class CustomDataset(Dataset):
  def __init__(self, data_path, transform = None):
    super(CustomDataset, self).__init__()
    self.image_data = sorted(glob(os.path.join(data_path, '*.jpg')))
    self.transform = transform
  
  def __getitem__(self, idx):
    image = PIL.Image.open(self.image_data[idx])
    transformed_image = self.transform(image)
    return transformed_image

  def __len__(self):
    return len(self.image_data)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min = 0.1, max = 2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size - 1

    def __call__(self, image):
        image = np.array(image)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)
        return image

class DataWrapper():
  def __init__(self, batch_size, validation_set_size, in_shape, data_path):
    self.batch_size = batch_size
    self.validation_set_size = validation_set_size
    self.in_shape = in_shape
    self.data_path = data_path
  
  def get_transformation(self):
    cl_jitter = transforms.ColorJitter(brightness = 0.7, contrast = 0.7, saturation = 0.7, hue = 0.2)
    transformation = transforms.Compose([transforms.RandomResizedCrop(size = self.in_shape[0], scale = (0.6, 1.0)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomApply([cl_jitter], p = 0.5),
                                         GaussianBlur(int(0.1 * self.in_shape[0])),
                                         transforms.ToTensor()])
    return transformation

  def get_dataset(self, path):
    dataset = CustomDataset(path, transform = ImageTransformation(self.get_transformation()))
    return dataset

  def get_loaders(self):
    dataset = self.get_dataset(self.data_path)
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    split = int(np.floor(self.validation_set_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_iterator = DataLoader(dataset, batch_size = self.batch_size, sampler = train_sampler, 
                                num_workers = 4, pin_memory = True, drop_last = True)
    val_iterator = DataLoader(dataset, batch_size = self.batch_size, sampler = val_sampler, 
                              num_workers = 4, pin_memory = True, drop_last = True)
    return train_iterator, val_iterator
