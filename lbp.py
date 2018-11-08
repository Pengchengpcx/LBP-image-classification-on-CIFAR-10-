import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data

import torchvision.transforms as transforms
import torchvision
import torch
import torchvision.datasets as dsets

METHOD = 'uniform'
radius = 1
n_points = 8 * radius

trans = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
training_set = dsets.SVHN(root='./data', split='train', transform=trans, download=True)
testing_set = dsets.SVHN(root='./data', split='test', transform=trans, download=True)


def lbp_vec(img, method, p, r, regions=4):
    '''
    :param img: input grayscale image
    :param method: LBP method
    :param p: number of neighbors
    :param r: radius
    :param regions: number of regions divided
    :return: lbp vector
    '''
    # crop the 32x32 image into 4 8x8 regions
    pieces = []
    for i in range(regions):
        for j in range(regions):
            pieces.append(img[8*i:8*(i+1), 8*j:8*(j+1)])

    for piece in pieces:
        lbp = local_binary_pattern(piece, p, r, METHOD)



    return True

train_lbp_vectors = []
train_labels = []



