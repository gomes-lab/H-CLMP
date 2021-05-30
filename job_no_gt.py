import os
from os import listdir
from os.path import isfile, join

'''
Pytorch implementation of the paper "Materials representation and transfer learning for multi-property prediction"
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu

This is an example script to train a model on uvis with ground truth and test the model on uvis without ground truth. 

For testing, please set train to be 0. The testing results will be saved in the "results" folder by defaults. We 
have provided trained models for our 69 systems. One can run the script to output results.
'''

model = 'run_HCLMP.py'
data_path = 'data/uvis_dataset_no_redundancy/uvis_dict.chkpt'
data_path_no_gt = 'data/uvis_dataset_no_gt/uvis_dict.chkpt' # The fake labels in this dataset are random numbers sampled from Uniform distribution.
# We create these fake labels just to have a uniform data format so that we don't need to revise codes of our model.
# You should only use the output prediction numpy file and ignore the fake label numpy file.
# The testing MAE is calculated with the fake labels, and just also can be ignored.

train = 1 # 0 for testing, 1 for training

transfer_type = 'gen_feat' # choices ['gen_feat', 'None']
#transfer_type = 'None'
epochs = 50

train_path = 'data/uvis_dataset_no_redundancy/idx/rd_idx/train/rd_idx.npy'
val_path = 'data/uvis_dataset_no_redundancy/idx/rd_idx/val/rd_idx.npy'
test_path = 'data/uvis_dataset_no_gt/idx/rd_idx.npy'


if train==1:
    command = "CUDA_VISIBLE_DEVICES=0 python %s --train --epochs %d --transfer-type %s --data-path %s --train-path %s --val-path %s"\
              %(model, epochs, transfer_type, data_path, train_path, val_path)
else:
    command = "CUDA_VISIBLE_DEVICES=0 python %s --evaluate --epochs %d --transfer-type %s --data-path %s --test-path %s"\
              %(model, epochs, transfer_type, data_path_no_gt, test_path)

print()
print(command)
print()
os.system(command)


