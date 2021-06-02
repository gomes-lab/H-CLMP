## Materials representation and transfer learning for multi-property prediction

Authors: Shufeng Kong <sup>1</sup>, Dan Guevarra <sup>2</sup>, Carla P. Gomes <sup>1</sup>, and John M. Gregoire <sup>2</sup>
1) Department of Computer Science, Cornell University, Ithaca, NY, USA
2) Division of Engineering and Applied Science, 
   California Institure of Technology, Pasadena, CA, USA
   
The hierarchical correlation learning for multi-property prediction (H-CLMP) framework combines a variational autoencoder, trained for multiple target properties, with a feature encoder from the material composition space to the same multi-property embedding space. A GAN-based transfer learning extension is also included, which learns an internal representation of compositions based on a different set of properties, such as one for which more data is available. A dataset is included for spectral optical absorption (the target properties) in 69 3-cation metal oxide composition spaces, as well as computed density of states (DOS) data from the Materials Project (the properties used for transfer learning).

This is a Pytorch implementation of the HCLMP model. 

### Datasets 

The spectral optical absorption dataset and a discription of the dataset can be found in https://data.caltech.edu/records/1878. To run our software, we need addtional DOS dataset from the material project https://materialsproject.org/. Therefore, we provide a colloection of all the necessary datasets to run our software in this link https://drive.google.com/drive/folders/1z5ULw7GcilB6L7Rjgkv5tOKL0n3xm0cT?usp=sharing. Please download the data.zip file and place in the main folder and unzip them.

### Enviroments

This software relies a number of packages:

Pytorch 1.7.1

Numpy 1.20.1

torch_scatter 2.0.6

json 2.0.9

tqdm 4.42.1

### Usages
1 and 2 can be ignored if you have downloaded our dataset from our provided Google drive lnk, we have done 
these for you. 

1). Pretrain a WGAN for transfer learning.

Change your current directory to the main folder and run:
```shell script
python WGAN_for_DOS/train_GAN.py
```

To test the quality of the generated DOS:
```shell script
python WGAN_for_DOS/test_GAN.py
```

2). Preprocess datasets

2.1 Copy the 'WGAN.py' and the pretrained WGAN model 'generator_MP2020.pt' to the 
'uvis_dataset_no_redundancy' folder.

Change your current directory  to the 'data/uvis_dataset_no_redundancy' folder and run:
```shell script
python process.py
```
This will generate train/test indices (validation set is a subset of the train set, see the 'data/uvis_dataset_no_redundancy/idx' folder) and 
a dictionary X. Each entry of X is a data entry, which is also a dictionary including keys: 

'fom': the material properties' label, representing the 10 dimension observation energy.

'composition_nonzero': the fraction of existing elements in the compound.

'composition_nonzero_idx': the corresponding indices of the existing elements 
in the array ['Ag', 'Ba', 'Bi', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'In', 'K', 'La', 'Mg', 'Mn', 'Mo', 'Nb',
             'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Rb', 'Sc', 'Sm', 'Sn', 'Sr', 'Tb', 'Ti', 'V', 'W', 'Yb', 'Zn', 'Zr'].

'nonzero_element_name': the names of the existing elements in the compound.

'gen_dos_fea': the generated DOS feature for the compound by using the pretrained WGAN
model.

'composition': the traction of all considered elements (['Ag', 'Ba', 'Bi', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'In', 'K', 'La', 'Mg', 'Mn', 'Mo', 'Nb',
             'Nd', 'Ni', 'P', 'Pb', 'Pd', 'Pr', 'Rb', 'Sc', 'Sm', 'Sn', 'Sr', 'Tb', 'Ti', 'V', 'W', 'Yb', 'Zn', 'Zr']) in the compound
 
2.2 Copy the 'WGAN.py' and the pretrained WGAN model 'generator_MP2020.pt' to the 
'data/uvis_dataset_no_gt' folder.

Change your current directory  to the 'data/uvis_dataset_no_gt' folder and run:
```shell script
python process.py
```

3). Example of training and testing the HCLMP model.

3.1 Change your current directory to the main folder and run:
```shell script
python jobs.py
```
This can be used to train/test the random split setting or the 69 ternary systems.

3.2 You can also run jobs in parallel (this needs to revise the script base on
your GPU numbers and memory. Read the script for details):
```shell script
bash jobs_parallel.sh
```
This can be used to train/test the 69 ternary systems in parallel.

3.3 To train a model on all labelled data and test the model on new data without ground-truth:
```shell script
python job_no_gt.py
```
This is for the experiment of "deployment for materials discovery" section in the paper.

3.4 We have provided our pretrained models, you can
set flag "train" to 0 and run the above scripts to test the models.

### Copyright

The graph encoder we adopted takes the element embedding and element fraction as input and produce an latent embedding.
This piece of codes is modified from Goodall and Lee, Predicting materials properties without crystal structure: 
deep representation learning from stoichiometry, Nature communication, 2020. Copyright of the graph encoder belongs to the authors. 
Please see the Github Link: https://github.com/CompRhys/roost for details.




