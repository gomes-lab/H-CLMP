import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.manual_seed(1024)
np.random.seed(1024)


'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

class Generator(nn.Module):
    def __init__(self, label_dim, feature_dim, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.laten_to_label = nn.Sequential(
            nn.Linear(feature_dim + latent_dim, 256),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(128),
            nn.Linear(128, label_dim),
            #nn.Sigmoid()
        )

    def forward(self, input_data):
        return self.laten_to_label(input_data)

    def sample_latent(self, num_sample):
        return torch.randn((num_sample, self.latent_dim))

class Discriminator(nn.Module):
    def __init__(self, label_dim, feature_dim):
        super(Discriminator, self).__init__()

        self.label_to_feature = nn.Sequential(
            nn.Linear(label_dim + feature_dim, 256),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )

    def forward(self, input_data):
        return self.label_to_feature(input_data)


