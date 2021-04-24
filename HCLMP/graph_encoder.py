import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch.utils.data import Dataset
import os
import json
import numpy as np
'''
This encoder takes the element embedding and element fraction as input and produce an latent embedding.
This piece of codes is taken from Goodall and Lee, Predicting materials properties without crystal structure: 
deep representation learning from stoichiometry, Nature communication, 2020. Copyright belongs to the authors. 
Please see the Github Link: https://github.com/CompRhys/roost for details.
'''

class Featuriser(object):
    """
    Base class for featurising nodes and edges.
    """

    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])


class LoadFeaturiser(Featuriser):
    """
    Initialize a featuriser from a JSON file.

    Parameters
    ----------
    embedding_file: str
        The path to the .json file
    """

    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super().__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """
    '''
    This class has been modified by us to fit our dataset.
    '''

    def __init__(self, data_path, fea_path, task):
        """
        """
        assert os.path.exists(data_path), "{} does not exist!".format(data_path)  # composition str and target
        # NOTE make sure to use dense datasets, here do not use the default na
        # as they can clash with "NaN" which is a valid material
        #self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])
        self.data = torch.load(data_path)

        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)  # element embedding
        self.elem_features = LoadFeaturiser(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size
        #print(self.elem_emb_len)
        self.task = task
        self.n_targets = len(self.data[0]['fom'])
        self.gen_feat_dim = len(self.data[0]['gen_dos_fea'])

    def get_target(self, index):
        tar_list = []
        for idx in index:
            tar = self.data[idx]['fom']
            tar_list.append(tar)
        tar = np.array(tar_list)
        return tar


    def __len__(self):
        return len(self.data)

    #@functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """

        Returns
        -------
        atom_weights: torch.Tensor shape (M, 1)
            weights of atoms in the material
        atom_fea: torch.Tensor shape (M, n_fea)
            features of atoms in the material
        self_fea_idx: torch.Tensor shape (M*M, 1)
            list of self indices
        nbr_fea_idx: torch.Tensor shape (M*M, 1)
            list of neighbor indices
        target: torch.Tensor shape (1,)
            target value for material
        cry_id: torch.Tensor shape (1,)
            input id for the material
        """
        #cry_id, composition, target = self.df.iloc[idx]
        cry_id = idx
        composition = self.data[idx]['nonzero_element_name']
        target = self.data[idx]['fom']
        gen_feat = self.data[idx]['gen_dos_fea']

        #elements, weights = parse_roost(composition)
        elements = self.data[idx]['nonzero_element_name']
        weights = self.data[idx]['composition_nonzero']
        if np.sum(weights) > 1.0+1e-5:
            print(weights)
            print(elements)
        assert np.sum(weights) <= 1.0+1e-5

        #weights = np.atleast_2d(weights).T / np.sum(weights)
        assert len(elements) != 1, f"cry-id {cry_id} [{composition}] is a pure system"
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights).unsqueeze(1)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)#.unsqueeze(1)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)#.unsqueeze(1)
        if self.task == "regression":
            targets = torch.Tensor([target]).squeeze()
            gen_feats = torch.Tensor([gen_feat]).squeeze()
        elif self.task == "classification":
            targets = torch.LongTensor([target]).squeeze()

        return (
            (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
            targets,
            gen_feats,
            composition,
            cry_id,
        )

def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_gen_feat = []
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, gen_feat, comp, cry_id) in enumerate(dataset_list):
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = inputs
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_gen_feat.append(gen_feat)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
        ),
        torch.stack(batch_target, dim=0),
        torch.stack(batch_gen_feat, dim=0),
        batch_comp,
        batch_cry_ids,
    )


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self, input_dim, output_dim, hidden_layer_dims, activation=nn.LeakyReLU,
        batchnorm=False
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1])
                                    for i in range(len(dims)-1)])
        else:
            self.bns = nn.ModuleList([nn.Identity()
                                    for i in range(len(dims)-1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x, index, weights):
        """ forward pass """

        gate = self.gate_nn(x)
        gate = gate - scatter_max(gate, index, dim=0)[0][index]

        if self.pow >0:
            gate = (weights ** self.pow) * gate.exp()
        else:
            gate = (1/(weights ** torch.abs(self.pow)+1e-10)) * gate.exp()

        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
        # self.embedding = nn.Linear(elem_emb_len, elem_fea_len)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

        # self.cry_pool = nn.ModuleList(
        #     [
        #         MeanPooling()
        #     ]
        # )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        #print(elem_fea.shape, elem_weights.shape)
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)
        #print(torch.sum(elem_fea))
        #print(torch.sum(elem_fea), torch.sum(elem_weights))
        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)
            #print(torch.sum(elem_fea), torch.sum(elem_weights))

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
                # attnhead(elem_fea, index=cry_elem_idx)
            )

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self):
        return self.__class__.__name__


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super().__init__()

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                )
                for _ in range(elem_heads)
            ]
        )

        # self.pooling = nn.ModuleList(
        #     [
        #         MeanPooling()
        #     ]
        # )
        # self.mean_msg = SimpleNetwork(2*elem_fea_len, elem_fea_len, elem_msg)

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
                # attnhead(self.mean_msg(fea), index=self_fea_idx)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea
        # return fea

    def __repr__(self):
        return self.__class__.__name__
