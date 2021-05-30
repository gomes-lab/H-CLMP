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
args.feature_dim = 39
args.latent_dim = 50
args.label_dim = 161
args.data_dir = './data/20200828_MP_dos/MP_comp39_dos161.npy'
args.train_idx = './data/20200828_MP_dos/train_idx.npy'
args.lr = 1e-4
args.betas = (.9, .99)
args.dataset = 'MP2020'

def train_GAN(args):
    print('Start training gan .... ')
    generator = Generator(label_dim=args.label_dim, latent_dim=args.latent_dim, feature_dim=args.feature_dim)
    discriminator = Discriminator(label_dim=args.label_dim, feature_dim=args.feature_dim)

    G_optimizer = optim.RMSprop(generator.parameters(), lr=5e-5)
    D_optimizer = optim.RMSprop(discriminator.parameters(), lr=5e-5)

    #lr = 1e-4
    #betas = (.9, .99)

    #G_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=args.betas)
    #D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=args.betas)

    trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, critic_iterations=2, use_cuda=torch.cuda.is_available())
    trainer.train(args=args, epochs=500)

    # Save models
    name = args.dataset
    torch.save(trainer.G.state_dict(), './WGAN_for_DOS/generator_' + name + '.pt')
    torch.save(trainer.D.state_dict(), './WGAN_for_DOS/discriminator_' + name + '.pt')

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=100,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'D_generated_loss': [], 'D_real_loss': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data

        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[0])

        # Calculate probabilities on real and generated data
        #data = Variable(data)
        if self.use_cuda:
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()

        d_x_real = torch.cat((data[0], data[1]), 1)
        d_x_generated = torch.cat((data[0], generated_data), 1)

        d_real = self.D(d_x_real)
        d_generated = self.D(d_x_generated)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(d_x_real, d_x_generated)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())
        self.losses['GP'].append(gradient_penalty.item())
        self.losses['D_generated_loss'].append(d_generated.mean().item())
        self.losses['D_real_loss'].append(d_real.mean().item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data[0].size()[0]
        generated_data = self.sample_generator(batch_size, data[0])

        # Calculate loss and optimize
        d_x_generated = torch.cat((data[0], generated_data), 1)
        d_generated = self.D(d_x_generated)

        #print(d_generated.mean())
        #exit()

        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        #alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def train_iteration(self, feat, label, current_step):
        self._critic_train_iteration([feat, label])
        if current_step % self.critic_iterations == 0:
            self._generator_train_iteration([feat, label])


    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

        print("D loss: {}".format(np.mean(self.losses['D'])))
        print("D gen loss: {}".format(np.mean(self.losses['D_generated_loss'])))
        print("D real loss: {}".format(np.mean(self.losses['D_real_loss'])))
        print("GP loss: {}".format(np.mean(self.losses['GP'])))
        print("Gradient norm: {}".format(np.mean(self.losses['gradient_norm'])))

        self.losses['D'].clear()
        self.losses['GP'].clear()
        self.losses['gradient_norm'].clear()
        self.losses['D_generated_loss'].clear()
        self.losses['D_real_loss'].clear()

        if self.num_steps > self.critic_iterations:
            print("G loss: {}".format(np.mean(self.losses['G'])))
            self.losses['G'].clear()
        print()

    def train(self, args, epochs):
        print("loading data.....")
        data = np.load(args.data_dir, allow_pickle=True).astype(float)
        print(data.shape)
        train_idx = np.load(args.train_idx)
        data = data[train_idx,:]
        print(data.shape)
        labels = torch.Tensor(data[:, args.feature_dim:]).cuda()
        features = torch.Tensor(data[:, 0:args.feature_dim]).cuda()

        mean = torch.mean(labels, dim=0).cuda()
        std = torch.std(labels, dim=0).cuda()
        np.save('label_mean.npy', mean.detach().cpu().numpy())
        np.save('label_std.npy', std.detach().cpu().numpy())
        #exit()

        labels_standard = (labels - mean) / (std + 1e-6)

        dataset = TensorDataset(features, labels_standard)


        data_loader = DataLoader(dataset, batch_size=128)
        print('finish loading data!')

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

    def sample_generator(self, num_samples, feature):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(torch.cat((feature, latent_samples), dim=1))
        return generated_data


train_GAN(args)
