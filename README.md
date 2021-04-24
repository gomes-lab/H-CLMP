## Materials representation and transfer learning for multi-property prediction

Authors: Shufeng Kong <sup>1</sup>, Dan Guevarra <sup>2</sup>, Carla P. Gomes <sup>1</sup>, and John M. Gregoire <sup>2</sup>
1) Department of Computer Science, Cornell University, Ithaca, NY, USA
2) Division of Engineering and Applied Science, 
   California Institure of Technology, Pasadena, CA, USA
   

This is a Pytorch implementation of the HCLMP model proposed in our paper. 

### Dataset, trained models, and results 

Datasets, trained models, and results can be found in this link https://drive.google.com/drive/folders/15_32k2iGQv-K5OUDrszJYEoBuI0Sb0Fk?usp=sharing
Please download the three zip files and place in the main folder and unzip them. A discription of the datasets
can be found in the paper.

### Enviroments

This software relies a number of packages:

Pytorch 1.7.1

Numpy 1.20.1

torch_scatter 2.0.6

json 2.0.9

tqdm 4.42.1

### Usages

1). Pretrain a WGAN for transfer learning.

Change the current working directory to the main folder and run:
```shell script
python WGAN_for_DOS/train_GAN.py
```

To test the quality of the generated DOS:
```shell script
python WGAN_for_DOS/test_GAN.py
```

2). Preprocess datasets

Copy the 'WGAN.py' and the pretrained WGAN model 'generator_MP2020.pt' to the 
'uvis_dataset_no_redundancy' folder.

Change the current working directory to the 'uvis_dataset_no_redundancy' folder and run:
```shell script
python process.py
```
This will process the csv dataset and generate train/test indices (validation set is a subset
of the train set) and 
transform the csv data set into a dictionary X. Each entry of X is
a data entry, which is also a dictionary including keys: 

'fom': the material properties' label, representing the 10 dimension observation energy.

'composition_nonzero': the fraction of existing elements in the compound.

'composition_nonzero_idx': the corresponding indices of the existing elements 
in the array ['Ag', 'Bi', 'Ca', 'Ce', 'Co', 'Cr', 'Cu', 'Er', 'Eu', 'Fe', 'Gd', 'K', 'La', 'Mg', 'Mn', 'Mo', 'Nb',
             'Nd', 'Ni', 'P', 'Pd', 'Pr', 'Sc', 'Sm', 'Sr', 'Ti', 'V', 'W', 'Yb', 'Zn', 'Zr'].

'nonzero_element_name': the names of the existing elements in the compound.

'gen_dos_fea': the generated DOS feature for the compound by using the pretrained WGAN
model.

'composition': the traction of all considered elements (['Ag', 'Bi', 'Ca', 'Ce', 'Co', 
'Cr', 'Cu', 'Er', 'Eu', 'Fe', 'Gd', 'K', 'La', 'Mg', 'Mn',
 'Mo', 'Nb', 'Nd', 'Ni', 'P', 'Pd', 'Pr', 'Sc', 'Sm', 'Sr', 
 'Ti', 'V', 'W', 'Yb', 'Zn', 'Zr']) in the compound
 
3). Example of training and testing the HCLMP model.

Change the current working directory to the main folder and run:
```shell script
python jobs.py
```

### Copyright

The graph encoder we adopted takes the element embedding and element fraction as input and produce an latent embedding.
This piece of codes is modified from Goodall and Lee, Predicting materials properties without crystal structure: 
deep representation learning from stoichiometry, Nature communication, 2020. Copyright of the graph encoder belongs to the authors. 
Please see the Github Link: https://github.com/CompRhys/roost for details.




