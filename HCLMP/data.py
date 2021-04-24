import numpy as np
import pandas as pd 
import functools
import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as torch_Dataset
from torch_geometric.data import Data, DataLoader as torch_DataLoader
import sys,json,os
from pymatgen.core.structure import Structure
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SPCL
import warnings
from HCLMP.utils import *
from os import path

#gpu_id = 0
#device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = set_device()

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class ELEM_Encoder:
    def __init__(self):
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'] # 103
        self.e_arr    = np.array(self.elements)

    def encode(self,composition_dict): # from formula to composition, which is a vector of length 103
        answer   = [0]*len(self.elements)

        elements = [str(i) for i in composition_dict.keys()]
        counts   = [j for j in composition_dict.values()]
        total    = sum(counts)

        for idx in range(len(elements)):
            elem   = elements[idx]
            ratio  = counts[idx]/total
            idx_e  = self.elements.index(elem)
            answer[idx_e] = ratio
        return torch.tensor(answer).float().view(1,-1)

    def decode_pymatgen_num(tensor_idx): # from ele_num to ele_name
        idx      = (tensor_idx-1).cpu().tolist()
        return self.e_arr[idx]
        

class DATA_normalizer:
    def __init__(self, array):
        tensor = torch.tensor(array)
        self.mean   = torch.mean(tensor, dim=0).float()
        self.std    = torch.std(tensor, dim=0).float()
    def reg(self,x):
        return x.float()

    def log10(self,x):
        return torch.log10(x)

    def delog10(self,x):
        return 10*x

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

class METRICS:
    def __init__(self,c_property,epoch,torch_criterion,torch_func,device):
        self.c_property        = c_property
        self.criterion         = torch_criterion
        self.eval_func         = torch_func
        self.dv                = device
        self.training_measure1 = torch.tensor(0.0).to(device)
        self.training_measure2 = torch.tensor(0.0).to(device)
        self.valid_measure1    = torch.tensor(0.0).to(device)
        self.valid_measure2    = torch.tensor(0.0).to(device)

        self.training_counter  = 0
        self.valid_counter     = 0

        self.training_loss1    = []
        self.training_loss2    = []
        self.valid_loss1       = []
        self.valid_loss2       = []
        self.duration          = []
        self.dataframe         = self.to_frame()

    def __str__(self):
        x = self.to_frame() 
        return x.to_string()

    def to_frame(self):
        metrics_df = pd.DataFrame(list(zip(self.training_loss1,self.training_loss2,
                                           self.valid_loss1,self.valid_loss2,self.duration)),
                                           columns=['training_1','training_2','valid_1','valid_2','time'])
        return metrics_df

    def set_label(self,which_phase,graph_data):
        use_label = graph_data.y
        return use_label

    def save_time(self,e_duration):
        self.duration.append(e_duration)


    def __call__(self,which_phase,tensor_pred,tensor_true,measure=1):
        if measure == 1:
            if   which_phase == 'training'  : 
                loss         = self.criterion(tensor_pred,tensor_true)
                self.training_measure1 += loss
            elif which_phase == 'validation':
                loss         = self.criterion(tensor_pred,tensor_true)
                self.valid_measure1    += loss
        else:
            if   which_phase == 'training'  :
                loss         = self.eval_func(tensor_pred,tensor_true) 
                self.training_measure2 += loss
            elif which_phase == 'validation':
                loss         = self.eval_func(tensor_pred,tensor_true)
                self.valid_measure2    += loss
        return loss
    def reset_parameters(self,which_phase,epoch):
        if   which_phase == 'training'  :
            # AVERAGES
            t1 = self.training_measure1/(self.training_counter)
            t2 = self.training_measure2/(self.training_counter)

            self.training_loss1.append(t1.item())
            self.training_loss2.append(t2.item())
            self.training_measure1 = torch.tensor(0.0).to(self.dv)
            self.training_measure2 = torch.tensor(0.0).to(self.dv)
            self.training_counter  = 0 
        else:
            # AVERAGES
            v1 = self.valid_measure1/(self.valid_counter)
            v2 = self.valid_measure2/(self.valid_counter)

            self.valid_loss1.append(v1.item())
            self.valid_loss2.append(v2.item())
            self.valid_measure1    = torch.tensor(0.0).to(self.dv)
            self.valid_measure2    = torch.tensor(0.0).to(self.dv)      
            self.valid_counter     = 0 
    def save_info(self):
        with open('MODELS/metrics_.pickle','wb') as metrics_file:
            pickle.dump(self,metrics_file)
            
class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step) # int((dmax-dmin) / step) + 1
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        #print(distances.shape) [nbr, nbr]
        #x = distances[..., np.newaxis] [nbr, nbr, 1]
        #print(self.filter.shape)
        #print((x-self.filter).shape)
        #exit()
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys()) # 100
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIF_Lister(Dataset):
    def __init__(self,crystals_ids,full_dataset,norm_obj=None,normalization=None,df=None,src=None):
        self.crystals_ids  = crystals_ids
        self.full_dataset  = full_dataset
        self.normalizer    = norm_obj
        self.material_ids  = df.iloc[crystals_ids].values[:,0].squeeze()  # MP-xxx
        self.normalization = normalization
        self.src           = src # e.g. NEW

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self,original_dataset):
        names    = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i        = self.crystals_ids[idx]
        material = self.full_dataset[i]

        n_features    = material[0][0]
        e_features    = material[0][1] # [n_atom, nbr, 41]
        e_features    = e_features.view(-1,41) if self.src in ['CGCNN','NEW'] else e_features.view(-1,9)
        #print(e_features.shape) # [n_atom * nbr, 41]
        a_matrix      = material[0][2]

        groups        = material[1]
        enc_compo     = material[2]  # normalize feat
        coordinates   = material[3]
        y             = material[4]  # target

        if   self.normalization == None:
            y = y

        elif self.normalization == 'log':
            y = self.normalizer.log10(y + 1e-3)

        elif self.normalization == 'classification-1':
            if   y == 0:    y = torch.tensor([0]).long()
            elif y == 1:    y = torch.tensor([1]).long()
        elif self.normalization == 'classification-0':
            if   y == 0:    y = torch.tensor([1]).long()
            elif y == 1:    y = torch.tensor([0]).long()

        graph_crystal = Data(x=n_features,y = y,edge_attr=e_features,edge_index=a_matrix,global_feature = enc_compo,\
                             cluster=groups,num_atoms=torch.tensor([len(n_features)]).float(),coords=coordinates,the_idx=torch.tensor([float(i)]))

        return graph_crystal

############## Graph augmentation dataset #############

class CIF_Lister_aug(Dataset):
    def __init__(self,crystals_ids,full_dataset,aug='none',aug_ratio=None):
        self.crystals_ids  = crystals_ids
        self.full_dataset  = full_dataset
        self.aug = aug
        self.aug_ratio = aug_ratio

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self,original_dataset):
        names    = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i        = self.crystals_ids[idx]
        material = self.full_dataset[i]
        n_features    = material[0][0]
        e_features    = material[0][1] # [n_atom, nbr, 41]
        e_features    = e_features.view(-1,41) # [n_atom * nbr, 41]
        a_matrix      = material[0][2]
        groups        = material[1]
        enc_compo     = material[2]  # normalize feat

        data = Data(x=n_features,edge_attr=e_features,edge_index=a_matrix,global_feature = enc_compo,
                    cluster=groups,the_idx=torch.tensor([float(i)]))

        if self.aug == 'dropN':
            data = drop_nodes(data, self.aug_ratio)
        elif self.aug == 'permE':
            data = permute_edges(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = mask_nodes(data, self.aug_ratio)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'random':
            n = np.random.randint(2)
            if n == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif n == 1:
                data = subgraph(data, self.aug_ratio)
                # data = subgraph(data, 0.5)
            else:
                print('augmentation error')
                assert False
        else:
            print('augmentation error')
            assert False
        return data


def drop_nodes(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def permute_edges(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)
    edge_index = data.edge_index.numpy()

    idx_add = np.random.choice(node_num, (2, permute_num))
    edge_index = np.concatenate((edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
    data.edge_index = torch.tensor(edge_index)

    return data


def mask_nodes(data, aug_ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data


def subgraph(data, aug_ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

    edge_index = data.edge_index.numpy()
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data

###########################################################

class CIF_Dataset(Dataset):
    def __init__(self, args, pd_data=None, np_data=None, norm_obj=None, normalization=None, max_num_nbr=12, radius=8, dmin=0, step=0.2, cls_num=3, root_dir ='DATA/CIF-DATA/', src=None):
        self.root_dir      = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.normalizer    = norm_obj  # not used
        self.normalization = normalization # not used
        self.pd_data       = pd_data
        self.np_data       = np_data
        self.ari           = AtomCustomJSONInitializer(self.root_dir+'atom_init.json')
        self.gdf           = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.clusterizer   = SPCL(n_clusters=cls_num, random_state=None,assign_labels='discretize')
        self.clusterizer2  = KMeans(n_clusters=cls_num, random_state=None)
        self.encoder_elem  = ELEM_Encoder()
        self.update_root   = None
        self.src = src
        #self.catche_data   = catche_data
        #self.catche_data_update = False
        self.args = args

    def __len__(self):
        return len(self.pd_data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        if self.src == 'NEW':
            cif_id = self.pd_data.iloc[idx][0]
            target = self.np_data[idx]
            #target = torch.Tensor(target.astype(float)).view(1,-1)
        else:
            cif_id, target = self.pd_data.iloc[idx]

        catche_data_exist = False
        if path.exists(f'./DATA/processed_data/' + cif_id + '.chkpt'):
            catche_data_exist = True

        if self.args.use_catached_data and catche_data_exist:
            #tmp_dist = self.catche_data[cif_id]
            tmp_dist = torch.load(f'./DATA/processed_data/' + cif_id + '.chkpt')
            atom_fea = tmp_dist['atom_fea']
            nbr_fea = tmp_dist['nbr_fea']
            nbr_fea_idx = tmp_dist['nbr_fea_idx']
            groups = tmp_dist['groups']
            enc_compo = tmp_dist['enc_compo']
            coordinates = tmp_dist['coordinates']
            target = tmp_dist['target']
            cif_id = tmp_dist['cif_id']
            atom_id = tmp_dist['atom_id']
            return (atom_fea, nbr_fea, nbr_fea_idx),groups,enc_compo,coordinates, target, cif_id, atom_id

        crystal = Structure.from_file(os.path.join(self.root_dir+'CIF/', cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])

        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True) # (site, distance, index, image)

        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] # [num_atom in this crystal]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        #print(nbr_fea_idx.shape) # [n_atom, nbr]
        #print(nbr_fea.shape) # [n_atom, nbr]
        nbr_fea     = self.gdf.expand(nbr_fea)
        #print(nbr_fea.shape) # [n_atom, nbr, 41]

        g_coords    = crystal.cart_coords
        #print(g_coords.shape) # [n_atom, 3]
        groups      = [0]*len(g_coords)
        if len(g_coords) > 2:
            try    : groups = self.clusterizer.fit_predict(g_coords)
            except : groups = self.clusterizer2.fit_predict(g_coords)
        groups  = torch.tensor(groups).long() # [n_atom]

        atom_fea     = torch.Tensor(atom_fea)
        nbr_fea      = torch.Tensor(nbr_fea)
        nbr_fea_idx  = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx)) # [2, E]
        #print(nbr_fea_idx.shape)
        #print(nbr_label_idx.shape)
        #exit()

        target       = torch.Tensor(target.astype(float)).view(1,-1)

        coordinates = torch.tensor(g_coords) # [n_atom, 3]
        enc_compo   = self.encoder_elem.encode(crystal.composition) # [1, 103]
        #print(crystal.composition)
        #print(target.shape)
        #print(coordinates.shape)
        #exit()

        #print(atom_fea.shape, nbr_fea.shape, nbr_fea_idx.shape, groups.shape, enc_compo.shape, coordinates.shape, target.shape, cif_id)
        tmp_dist = {}
        tmp_dist['atom_fea'] = atom_fea
        tmp_dist['nbr_fea'] = nbr_fea
        tmp_dist['nbr_fea_idx'] = nbr_fea_idx
        tmp_dist['groups'] = groups
        tmp_dist['enc_compo'] = enc_compo
        tmp_dist['coordinates'] = coordinates
        tmp_dist['target'] = target
        tmp_dist['cif_id'] = cif_id
        tmp_dist['atom_id'] = [crystal[i].specie for i in range(len(crystal))]

        #if cif_id not in self.catche_data.keys():
        #    self.catche_data[cif_id] = tmp_dist
        #    self.catche_data_update = True
        #    #torch.save(self.catche_data, "cache_data.chkpt")

        return (atom_fea, nbr_fea, nbr_fea_idx),groups,enc_compo,coordinates, target, cif_id,[crystal[i].specie for i in range(len(crystal))]

    def format_adj_matrix(self,adj_matrix):
        size           = len(adj_matrix)
        src_list       = list(range(size))
        all_src_nodes  = torch.tensor([[x]*adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes  = adj_matrix.view(-1).unsqueeze(0)

        return torch.cat((all_src_nodes,all_dst_nodes),dim=0)
