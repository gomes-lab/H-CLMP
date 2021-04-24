import pandas as pd, os.path as path
import numpy as np
import os

setting_folder=r'./results/'

fns=os.listdir(setting_folder)

transfer_type = '_gen_feat'

all_label = []
all_pred = []

for item in fns:
    pred = np.load(setting_folder + item + '/pred' + transfer_type + '.npy', allow_pickle=True)
    label = np.load(setting_folder + item + '/label' + transfer_type + '.npy', allow_pickle=True)
    all_pred.append(pred)
    all_label.append(label)

#print(len(all_label))

all_label = np.concatenate(all_label, axis=0)
all_pred = np.concatenate(all_pred, axis=0)

print(all_label.shape)
print(all_pred.shape)

print(np.mean(np.abs(all_pred-all_label), axis=0))
print(np.mean(np.abs(all_pred-all_label)))
