import enum
from numpy import insert
from sqlalchemy import all_
from torch_geometric.data import Data, Dataset
from sympy.combinatorics import Permutation, PermutationGroup

from etflow.commons.featurization import MoleculeFeaturizer

from .geom import GEOM

import pynauty

import torch
import numpy as np
import time 
from collections import defaultdict
import random
from collections import deque
import pickle
from functools import lru_cache 

class EuclideanDataset(Dataset):
    """Returns 3D Graph for different datasets

    Usage
    -----
    ```python
    from etflow.data import EuclideanDataset
    # pass path to processed data_dir
    dataset = EuclideanDataset(data_dir=<>)
    ```
    """

    def __init__(
        self,
        data_dir: str,
        use_ogb_feat: bool = False,
        use_edge_feat: bool = False,
        train_mode = "baseline",
        max_perms: int = 50,
    ):
        super().__init__()
        # instantiate dataset
        self.dataset = GEOM(data_dir=data_dir)
        self.mol_feat = MoleculeFeaturizer()
        self.use_ogb_feat = use_ogb_feat
        self.use_edge_feat = use_edge_feat
        self.cache = {}
        self.max_perm = max_perms   
        self.train_mode = train_mode
        if train_mode == "min":
            self.train_mode = "ief"

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data_bunch = self.dataset[idx]

        # get positions, atomic_numbers and smiles
        atomic_numbers = data_bunch["atomic_numbers"]
        pos = data_bunch["pos"]
        smiles = data_bunch["smiles"] # np.str
        sampled_pos = pos.clone()

        # featurize molecule
        node_attr = self.mol_feat.get_atom_features(smiles, self.use_ogb_feat)
        if node_attr.shape[0] != pos.shape[0]:
            breakpoint()

        chiral_index, chiral_nbr_index, chiral_tag = self.mol_feat.get_chiral_centers(smiles)
        edge_index, edge_attr = self.mol_feat.get_edge_index(smiles, use_edge_feat=self.use_edge_feat)
        num_edges = edge_index.shape[1]
        edge_index_forward = edge_index[:, :num_edges // 2]
        mol = self.mol_feat.get_mol_with_conformer(smiles, pos)
        
        num_nodes = len(atomic_numbers)
        assert node_attr.shape[0] == num_nodes
        if self.train_mode == "ief" and self.max_perm > 1:

            node_features = node_attr
                            
            graph = pynauty.Graph(number_of_vertices=num_nodes, directed=False)

            for i in range(edge_index_forward.shape[1]):
                graph.connect_vertex(edge_index_forward[0, i].item(), [edge_index_forward[1, i].item()])

            color_groups = defaultdict(set)
            for node in range(num_nodes):
                color =  tuple(node_features[node].cpu().numpy().tolist())
                color_groups[color].add(node)             
            vertex_colors = list(color_groups.values())

            graph.set_vertex_coloring(vertex_colors)
                    
            generators, _, _, _, _ = pynauty.autgrp(graph)

            # Convert generators to sympy Permutations

            generators_sympy = [Permutation(g) for g in generators]
            random.shuffle(generators_sympy)
            # Construct the permutation group
            aut_group = PermutationGroup(generators_sympy)

            # Print all isomorphisms (automorphisms)
            all_perms = [np.arange(num_nodes)]
            non_hydrogen_mask = atomic_numbers != 1

            for perm in aut_group.generate():
                if len(perm.array_form) == 0:
                    continue
                perm_array = np.array(perm.array_form)
                moved = np.where(perm_array != np.arange(len(perm_array)))[0]
    
                signature = moved[non_hydrogen_mask[moved]]
                if len(signature) == 0:
                    continue
                all_perms.append(perm.array_form)
                if len(all_perms) >= self.max_perm:
                    break
        
            num_perms = len(all_perms)
            if num_perms == 0:
                all_perms = torch.full((self.max_perm, num_nodes), -1, dtype=torch.long)
                all_perms[0] = torch.arange(num_nodes)
            else:
                all_perms = torch.stack([torch.tensor(perm, dtype=torch.long) for perm in all_perms])
                num_perms = all_perms.size(0)

                if num_perms < self.max_perm:
                    padding = torch.full((self.max_perm - num_perms, num_nodes), -1, dtype=torch.long)
                    all_perms = torch.cat([all_perms, padding], dim=0)

        else:
            all_perms = torch.zeros(self.max_perm, num_nodes).long()
            all_perms[0] = torch.arange(num_nodes)

        assert ((all_perms - torch.arange(num_nodes).unsqueeze(0)).abs().sum(-1) == 0).sum() == 1

        graph = Data(
            pos=pos,
            atomic_numbers=atomic_numbers,
            smiles=smiles,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            mol=mol,
            node_attr=node_attr,
            edge_attr=edge_attr if self.use_edge_feat else None,
            all_perms=all_perms.t(),
            all_pos=sampled_pos,
        )
        return graph
