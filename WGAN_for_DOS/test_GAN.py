import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.data import TensorDataset, DataLoader
from WGAN import Generator, Discriminator

'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Object(object):
    pass

args = Object()
args.feature_dim = 31
args.latent_dim = 50
args.label_dim = 161

args.data_dir = './data/20200828_MP_dos/MP_comp31_dos161.npy'
args.test_idx = './data/20200828_MP_dos/test_idx.npy'

args.mean_path = './WGAN_for_DOS/label_mean.npy'
args.std_path = './WGAN_for_DOS/label_std.npy'

args.lr = 1e-4
args.betas = (.9, .99)
args.dataset = 'MP2020'

data = np.load(args.data_dir,allow_pickle=True).astype(np.float32)
#train_idx = np.load(args.train_idx,allow_pickle=True).astype(int)
test_idx = np.load(args.test_idx,allow_pickle=True).astype(int)
mean = torch.from_numpy(np.load(args.mean_path))
std = torch.from_numpy(np.load(args.std_path))

test_feat = torch.from_numpy(data[test_idx,:][:,0:31])
test_label = torch.from_numpy(data[test_idx,:][:,31:])
#print(test_feat.shape, test_label.shape)
test_label_mean = torch.mean(test_label, dim=0)
test_label_std = torch.std(test_label, dim=0)
batch_size = test_label.shape[0]


G = Generator(label_dim=args.label_dim, latent_dim=args.latent_dim, feature_dim=args.feature_dim)
G.load_state_dict(torch.load('./WGAN_for_DOS/generator_MP2020.pt'))
G.eval()

def sample_generator(G, num_samples, feature):
    generated_data_all = 0
    num_sam = 100
    for i in range(num_sam):
        latent_samples = Variable(G.sample_latent(num_samples))
        latent_samples = latent_samples
        generated_data = G(torch.cat((feature, latent_samples), dim=1))
        generated_data_all += generated_data
    generated_data = generated_data_all/num_sam
    return generated_data

print('testing...')
gen_data = sample_generator(G, batch_size, test_feat).detach()
print('done!')
#gen_data = gen_data*std+mean
test_label = (test_label-mean)/std

MAE = torch.mean(torch.abs(gen_data-test_label))
print('MAE:', MAE.numpy())

# save testing results
np.save('./WGAN_for_DOS/test_label.npy', test_label)
np.save('./WGAN_for_DOS/test_pred.npy', gen_data)




