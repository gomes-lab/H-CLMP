import numpy as np
import torch
import random
from tqdm import tqdm
from HCLMP.HCLMP import HCLMP, compute_loss
from torch.utils.data import Dataset, DataLoader
import os
import json
from HCLMP.graph_encoder import CompositionData, collate_batch

'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu
'''

class MyDataset(Dataset):
    def __init__(self, Data, idx_path):
        self.data = Data
        self.idx = np.load(idx_path)

    def __len__(self):
        return len(self.idx)

    def get_all_target(self):
        all_target = []
        for i in self.idx:
            item = self.data[i]
            all_target.append(np.expand_dims(item['fom'], axis=0))
        all_target = np.concatenate(all_target, axis=0)
        return torch.as_tensor(all_target)

    def __getitem__(self, idx):
        item = self.data[self.idx[idx]]
        y = item['fom']
        ele_comp = item['composition']
        gen_feat = item['gen_dos_fea']

        return (torch.as_tensor(ele_comp), torch.as_tensor(y), torch.as_tensor(gen_feat))


class Scaler():
    def fit(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def scale(self, data):
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# train and save models
def train(args):
    print(f'\nTraining on sys {args.sys_name} \n')
    Data = torch.load(args.data_path)
    dataset = MyDataset(Data, args.train_path)
    all_target = dataset.get_all_target().to(args.device)
    MyScaler = Scaler()
    MyScaler.fit(all_target)
    #MyScaler.mean = MyScaler.mean.to(args.device)
    #MyScaler.std = MyScaler.std.to(args.device)

    #composition_dataset = CompositionData(args.data_path, "data/embeddings/megnet16-embedding.json", "regression")
    #composition_dataset = CompositionData(args.data_path, "data/embeddings/cgcnn-embedding.json", "regression")
    composition_dataset = CompositionData(args.data_path, "data/embeddings/matscholar-embedding.json", "regression")
    train_idx = np.load(args.train_path)
    val_idx = np.load(args.val_path)
    train_dataset = torch.utils.data.Subset(composition_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(composition_dataset, val_idx)

    train_loader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn= collate_batch)

    val_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn= collate_batch)

    elem_emb_len = composition_dataset.elem_emb_len

    model = HCLMP(args.feat_dim, args.label_dim, args.transfer_type, args.gen_feat_dim, elem_emb_len, args.device).to(args.device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-2)
    one_epoch_iter = np.ceil(len(train_dataset) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, one_epoch_iter * (args.epochs / args.decay_times), args.decay_ratios)

    best_loss = 1e+10
    for epoch in range(args.epochs):
        # train
        model.train()
        total_loss_smooth = 0
        nll_loss_e_smooth = 0
        nll_loss_x_smooth = 0
        kl_loss_smooth = 0
        count = 0
        pred_e = []
        pred_x = []
        label = []
        for input_, y, gen_feat, _, _ in tqdm(train_loader, mininterval=0.5, desc='(Training)', position=0, leave=True, ascii=True):
            input_ = (tensor.to(args.device) for tensor in input_)
            y = y.to(args.device)
            y_norm = MyScaler.scale(y)
            gen_feat = gen_feat.to(args.device)

            out = model(y_norm, gen_feat, *input_)
            total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            pred_e.append(out['label_out'])
            pred_x.append(out['feat_out'])
            label.append(y_norm)
            total_loss_smooth += total_loss
            nll_loss_e_smooth += nll_loss_e
            nll_loss_x_smooth += nll_loss_x
            kl_loss_smooth += kl_loss
            count += 1

        pred_e = torch.cat(pred_e, dim=0)
        pred_x = torch.cat(pred_x, dim=0)
        label = torch.cat(label, dim=0)
        total_loss_smooth = total_loss_smooth / count
        nll_loss_e_smooth = nll_loss_e_smooth / count
        nll_loss_x_smooth = nll_loss_x_smooth / count
        kl_loss_smooth = kl_loss_smooth / count
        lr = optimizer.param_groups[0]['lr']

        print("\n********** TRAINING STATISTIC ***********")
        print("epoch =%.1f\t lr =%.6f\t total_loss =%.6f\t nll_loss_e =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t" %
              (epoch, lr, total_loss_smooth, nll_loss_e_smooth, nll_loss_x_smooth, kl_loss_smooth))
        #print("mae=%.6f\t mae_x=%.6f\t mae_ori=%.6f\t mae_x_ori=%.6f" % (mae, mae_x, mae_ori, mae_x_ori))
        print("\n*****************************************")


        # validate
        model.eval()
        total_loss_smooth = 0
        nll_loss_e_smooth = 0
        nll_loss_x_smooth = 0
        kl_loss_smooth = 0
        count = 0
        pred_e = []
        pred_x = []
        label = []
        for input_, y, gen_feat, _, _ in tqdm(val_loader, mininterval=0.5, desc='(Validating)', position=0, leave=True, ascii=True):
            input_ = (tensor.to(args.device) for tensor in input_)
            y = y.to(args.device)
            y_norm = MyScaler.scale(y)
            gen_feat = gen_feat.to(args.device)

            with torch.no_grad():
                out = model(y_norm, gen_feat, *input_)
                total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

            pred_e.append(out['label_out'])
            pred_x.append(out['feat_out'])
            label.append(y_norm)
            total_loss_smooth += total_loss
            nll_loss_e_smooth += nll_loss_e
            nll_loss_x_smooth += nll_loss_x
            kl_loss_smooth += kl_loss
            count += 1

        pred_e = torch.cat(pred_e, dim=0)
        pred_x = torch.cat(pred_x, dim=0)
        label = torch.cat(label, dim=0)
        total_loss_smooth = total_loss_smooth / count
        nll_loss_e_smooth = nll_loss_e_smooth / count
        nll_loss_x_smooth = nll_loss_x_smooth / count
        kl_loss_smooth = kl_loss_smooth / count

        print("\n********** VALIDATING STATISTIC ***********")
        print("epoch =%.1f\t total_loss =%.6f\t nll_loss_e =%.6f\t nll_loss_x =%.6f\t kl_loss =%.6f\t" %
              (epoch, total_loss_smooth, nll_loss_e_smooth, nll_loss_x_smooth, kl_loss_smooth))
        print("\n*****************************************")

        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['scaler_state'] = MyScaler.state_dict()
        checkpoint['args'] = args
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, args.save_path + 'checkpoint_' + args.transfer_type + '.pth.tar')
        if best_loss > nll_loss_x_smooth:
            best_loss = nll_loss_x_smooth
            torch.save(checkpoint, args.save_path + 'best_' + args.transfer_type + '.pth.tar')


# load and test models, as well as saving results
def test(args):
    print(f'\nTesting on sys {args.sys_name} \n')
    path = args.save_path + 'best_' + args.transfer_type + '.pth.tar'
    print('checkpoint path:', path)
    checkpoint = torch.load(path, map_location=args.device)
    args_save = checkpoint['args']
    args.feat_dim = args_save.feat_dim
    args.gen_feat_dim = args_save.gen_feat_dim
    args.label_dim = args_save.label_dim

    MyScaler = Scaler()
    MyScaler.load_state_dict(checkpoint['scaler_state'])
    Data = torch.load(args.data_path)

    #composition_dataset = CompositionData(args.data_path, "data/embeddings/megnet16-embedding.json", "regression")
    #composition_dataset = CompositionData(args.data_path, "data/embeddings/cgcnn-embedding.json", "regression")
    composition_dataset = CompositionData(args.data_path, "data/embeddings/matscholar-embedding.json", "regression")
    test_idx = np.load(args.test_path)
    test_dataset = torch.utils.data.Subset(composition_dataset, test_idx)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             collate_fn= collate_batch)

    elem_emb_len = composition_dataset.elem_emb_len

    model = HCLMP(args.feat_dim, args.label_dim, args.transfer_type, args.gen_feat_dim, elem_emb_len, args.device).to(args.device)
    model.load_state_dict(checkpoint['model'])

    # test
    model.eval()
    total_loss_smooth = 0
    nll_loss_e_smooth = 0
    nll_loss_x_smooth = 0
    kl_loss_smooth = 0
    count = 0
    pred_e = []
    pred_x = []
    label = []

    for input_, y, gen_feat, _, _ in tqdm(test_loader, mininterval=0.5, desc='(Testing)', position=0, leave=True, ascii=True):
        input_ = (tensor.to(args.device) for tensor in input_)
        y = y.to(args.device)
        y_norm = MyScaler.scale(y)
        gen_feat = gen_feat.to(args.device)

        with torch.no_grad():
            out = model(y_norm, gen_feat, *input_)
            total_loss, nll_loss_e, nll_loss_x, kl_loss = compute_loss(y_norm, out)

        pred_e.append(out['label_out'])
        pred_x.append(out['feat_out'])
        label.append(y_norm)
        total_loss_smooth += total_loss
        nll_loss_e_smooth += nll_loss_e
        nll_loss_x_smooth += nll_loss_x
        kl_loss_smooth += kl_loss
        count += 1

    pred_e = torch.cat(pred_e, dim=0)
    pred_x = torch.cat(pred_x, dim=0)
    label = torch.cat(label, dim=0)
    total_loss_smooth = total_loss_smooth / count
    nll_loss_e_smooth = nll_loss_e_smooth / count
    nll_loss_x_smooth = nll_loss_x_smooth / count
    kl_loss_smooth = kl_loss_smooth / count

    nll_loss_e = torch.mean(torch.abs(pred_e-label))
    nll_loss_x = torch.mean(torch.abs(pred_x-label))

    print("\n********** TESTING STATISTIC ***********")
    print("epoch =%.1f\t total_loss =%.6f\t new_nll_loss_e =%.6f\t new_nll_loss_x =%.6f\t kl_loss =%.6f\t" %
          (checkpoint['epoch'], total_loss_smooth, nll_loss_e, nll_loss_x, kl_loss_smooth))
    print("\n*****************************************")

    result_path = f"results/{args.sys_name}/"
    label = label.data.cpu().numpy()
    pred = pred_x.data.cpu().numpy()
    mean = MyScaler.mean.data.cpu().numpy()
    std = MyScaler.std.data.cpu().numpy()

    np.save(result_path + 'pred_' + args.transfer_type + '.npy', pred)
    np.save(result_path + 'label_' + args.transfer_type + '.npy', label)
    np.save(result_path + 'mean_' + args.transfer_type + '.npy', mean)
    np.save(result_path + 'std_' + args.transfer_type + '.npy', std)














