import os
from os import listdir
from os.path import isfile, join

'''
Author: Shufeng KONG, Cornell University, USA
Contact: sk2299@cornell.edu

This is an example script to run jobs. Set single_job to be True if you only have one setting or dataset to run.
In our experiments, we have 69 systems to run, so we set single_job to be False by default. The trained models will be
saved in the "models" folder by default.

For testing, please set train to be 0. The testing results will be saved in the "results" folder by defaults. We 
have provided trained models for our 69 systems. One can run the script to output results.

The transfer_type indicates whether to use the GAN transfer learning. 'None' represents no transfer learning is used.
'''

model = 'run_HCLMP.py'
data_path = 'data/uvis_dataset_no_redundancy/uvis_dict.chkpt'

single_job = True
train = 0 # 0 for testing, 1 for training

transfer_type = 'gen_feat' # choices ['gen_feat', 'None']
#transfer_type = 'None'
epochs = 40

# Run on the ramdom split setting
if single_job:
    train_path = 'data/uvis_dataset_no_redundancy/idx/rd_idx_jh/train/rd_idx_jh.npy'
    test_path = 'data/uvis_dataset_no_redundancy/idx/rd_idx_jh/test/rd_idx_jh.npy'
    val_path = 'data/uvis_dataset_no_redundancy/idx/rd_idx_jh/val/rd_idx_jh.npy'

    if train==1:
        command = "CUDA_VISIBLE_DEVICES=0 python %s --train --epochs %d --transfer-type %s --data-path %s --train-path %s --val-path %s"\
                  %(model, epochs, transfer_type, data_path, train_path, val_path)
    else:
        command = "CUDA_VISIBLE_DEVICES=0 python %s --evaluate --epochs %d --transfer-type %s --data-path %s --test-path %s"\
                  %(model, epochs, transfer_type, data_path, test_path)

    print()
    print(command)
    print()
    os.system(command)
    
# Run on 69 ternary systems
else:
    train_dir = 'data/uvis_dataset_no_redundancy/idx/train/'
    val_dir = 'data/uvis_dataset_no_redundancy/idx/val_from_train/'
    test_dir = 'data/uvis_dataset_no_redundancy/idx/test/'

    system_files = sorted([f.split('.')[0] for f in listdir(train_dir) if isfile(join(train_dir, f))])

    for sys in system_files:
        train_path = train_dir + sys + '.npy'
        val_path = val_dir + sys + '.npy'
        test_path = test_dir + sys + '.npy'
        
        if train==1:
            command = "CUDA_VISIBLE_DEVICES=0 python %s --train --epochs %d --transfer-type %s --data-path %s --train-path %s --val-path %s"\
                      %(model, epochs, transfer_type, data_path, train_path, val_path)
        else:
            command = "CUDA_VISIBLE_DEVICES=0 python %s --transfer-type %s --evaluate --epochs %d --data-path %s --test-path %s" \
                      % (model, transfer_type, epochs, data_path, test_path)

        print()
        print(command)
        print()
        os.system(command)

    print('Finish running all systems!!!')

