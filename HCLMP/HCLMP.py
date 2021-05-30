import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn
from random import sample
from copy import copy, deepcopy
#from HCLMP.utils import *
from HCLMP.graph_encoder import DescriptorNetwork

'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

kldivloss = torch.nn.KLDivLoss(reduction='batchmean')

class HCLMP(nn.Module):
    def __init__(self, input_dim, label_dim, transfer_type, gen_feat_dim, elem_emb_len, device):
        super(HCLMP, self).__init__()
        input_dim = 64
        self.latent_dim = 128
        self.emb_size = 512
        self.label_dim = label_dim
        self.scale_coeff = 1.0
        self.transfer_type = transfer_type
        self.keep_prob = 0.0
        self.label_correlation = True
        self.device = device

        if transfer_type == 'gen_feat':
            #print(f'using transfer type {transfer_type}\n')
            self.input_dim = input_dim*2 #+ gen_feat_dim
        else:
            #print(f'using transfer type {transfer_type}\n')
            self.input_dim = input_dim

        self.fx1 = nn.Linear(self.input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, self.latent_dim)
        self.fx_logvar = nn.Linear(256, self.latent_dim)

        self.fd_x1 = nn.Linear(self.input_dim + self.latent_dim, 512)
        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size)
        )

        # label layers
        self.fe0 = nn.Linear(self.label_dim, self.emb_size) # shared label embedding
        self.fe1 = nn.Linear(self.label_dim, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, self.latent_dim)
        self.fe_logvar = nn.Linear(256, self.latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2

        self.bias = nn.Parameter(torch.zeros(self.label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        self.gen_feat_map = nn.Linear(gen_feat_dim, input_dim)
        self.graph_encoder = DescriptorNetwork(elem_emb_len)

        # things they share
        self.dropout = nn.Dropout(p=self.keep_prob)
        # label attention graph
        encoder_layer = nn.TransformerEncoderLayer(self.emb_size, nhead=4, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_after_atten = nn.Linear(self.emb_size,1, bias=False)
        self.out_after_atten_x = nn.Linear(self.emb_size, 1, bias=False)
        # label covariance matrix
        r_sqrt_sigma = nn.Parameter(torch.from_numpy(np.random.uniform(-np.sqrt(6.0/(self.label_dim+self.label_dim)), np.sqrt(6.0/(self.label_dim+self.label_dim)), (self.label_dim, int(self.label_dim)))).float())
        self.register_parameter("r_sqrt_sigma", r_sqrt_sigma)


    def label_encode(self, x):
        h1 = self.dropout(F.relu(self.fe1(x)))  # [label_dim, 512]
        h2 = self.dropout(F.relu(self.fe2(h1)))  # [label_dim, 256]
        mu = self.fe_mu(h2) * self.scale_coeff  # [label_dim, latent_dim]
        logvar = self.fe_logvar(h2) * self.scale_coeff  # [label_dim, latent_dim]

        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff  # [bs, latent_dim]
        logvar = self.fx_logvar(h3) * self.scale_coeff
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }
        return fx_output

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def label_decode(self, z):
        d1 = F.relu(self.fd1(z))
        d2 = F.leaky_relu(self.fd2(d1))
        return d2

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z))
        d2 = F.leaky_relu(self.fd_x2(d1))
        return d2

    def label_forward(self, label, feat):
        fe_output = self.label_encode(label)
        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']
        z = self.label_reparameterize(mu, logvar)
        label_emb = self.label_decode(torch.cat((feat, z), 1)) # [bs, emb_size]
        fe_output['label_emb'] = label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']  # [bs, latent_dim]
        logvar = fx_output['fx_logvar']  # [bs, latent_dim]
        z = self.feat_reparameterize(mu, logvar)  # [bs, latent_dim]
        feat_emb = self.feat_decode(torch.cat((x, z), 1))  # [bs, emb_size]
        fx_output['feat_emb'] = feat_emb
        return fx_output

    #def forward(self, label, gen_feat, feat):
    def forward(self, label, gen_feat, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):

        # We use the graph attention neural network from the Roost model to encoder element composition features.
        # Please refer to the work of Goodall and Lee, natural communication, 2020.
        feat = self.graph_encoder(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx
        ) # (bs, 64)

        if self.transfer_type == 'gen_feat':
            gen_feat = self.gen_feat_map(gen_feat)
            #gen_feat = F.normalize(gen_feat, dim=1)
            feat = torch.cat((feat, gen_feat), dim=1)

        fe_output = self.label_forward(label, feat)
        label_emb = fe_output['label_emb'] # [bs, emb_size]

        fx_output = self.feat_forward(feat)
        feat_emb  = fx_output['feat_emb'] # [bs, emb_size]
        embs = self.fe0.weight  # [emb_size, label_dim]

        if not self.label_correlation:
            label_out = torch.matmul(label_emb, embs)  # [bs, emb_size] * [emb_size, label_dim] = [bs, label_dim]
            feat_out = torch.matmul(feat_emb, embs)  # [bs, label_dim]
        else:
            label_out0 = torch.matmul(label_emb, embs)  # [bs, emb_size] * [emb_size, label_dim] = [bs, label_dim]
            feat_out0 = torch.matmul(feat_emb, embs)  # [bs, label_dim]

            # generate label embedding from label multivariate Gaussian, perform global label correlation learning
            noise = torch.normal(0, 1, size=(embs.shape[0], label_out0.shape[0], int(label_out0.shape[1]))).to(self.device) # [emb_size, bs, label_dim]
            B = self.r_sqrt_sigma.T.float().to(self.device) # [label_dim, label_dim]
            label_emb = torch.tensordot(noise, B, dims=1) + label_out0
            feat_emb = torch.tensordot(noise, B, dims=1) + feat_out0 #  [emb_size, bs, label_dim]
            label_emb = F.normalize(label_emb, dim=0) #  [emb_size, bs, label_dim]
            feat_emb = F.normalize(feat_emb, dim=0) #  [emb_size, bs, label_dim]
            label_out = label_emb.permute(1, 2, 0) #  [bs, label_dim, emb_size]
            feat_out = feat_emb.permute(1, 2, 0) #  [bs, label_dim, emb_size]

            # run label attention graph, perform conditional higher-order label correlation learning
            label_out = label_out.transpose(0, 1) #  [label_dim, bs, emb_size]
            feat_out = feat_out.transpose(0, 1) #  [label_dim, bs, emb_size]
            label_out = self.transformer_encoder(label_out)
            feat_out = self.transformer_encoder(feat_out)
            label_out = label_out.transpose(0, 1) #  [bs, label_dim, emb_size]
            feat_out = feat_out.transpose(0, 1) #  [bs, label_dim, emb_size]
            label_out = self.out_after_atten(label_out).squeeze()*0.5 + label_out0
            feat_out = self.out_after_atten_x(feat_out).squeeze()*0.5 + feat_out0

        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs
        output['label_out'] = label_out
        output['feat_out'] = feat_out
        return output

def compute_loss(input_label, output):
    fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb']
    fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb']

    # KL loss for model alignment
    kl_loss = torch.mean(0.5 * torch.sum(
        (fx_logvar - fe_logvar) - 1 + torch.exp(fe_logvar - fx_logvar) + torch.square(fx_mu - fe_mu) / (
                    torch.exp(fx_logvar) + 1e-6), dim=1))

    # KL loss for regularization
    fe_out_dis = torch.nn.functional.softmax(fe_out, dim=1)
    fx_out_dis = torch.nn.functional.softmax(fx_out, dim=1)
    label_dis = torch.nn.functional.softmax(input_label, dim=1)
    fe_kl = kldivloss(fe_out_dis, label_dis)
    fx_kl = kldivloss(fx_out_dis, label_dis)

    # L1 loss
    nll_loss = torch.mean(torch.abs((fe_out-input_label)))
    nll_loss_x = torch.mean(torch.abs((fx_out-input_label)))

    total_loss = (nll_loss + nll_loss_x) * 1 + kl_loss * 0.1 + (fe_kl + fx_kl) * 1

    return total_loss, nll_loss, nll_loss_x, kl_loss







