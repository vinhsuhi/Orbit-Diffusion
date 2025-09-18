import hydra
import omegaconf
import torch
import pandas as pd
from collections import defaultdict
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)

import itertools
from math import factorial
from itertools import permutations, product, islice

def compute_total_permutations(group_indices):
    """ Compute total possible full permutations by taking the product of factorials per group. """
    return np.prod([factorial(len(indices)) for indices in group_indices.values()])


def generate_permutations(atom_types, length, n_perms):
    unique_types = np.unique(atom_types)
    np.random.shuffle(unique_types)
    indices_by_type = {t: np.where(atom_types == t)[0] for t in unique_types if len(np.where(atom_types == t)[0]) > 1}
    permuted_indices = []

    for _ in range(n_perms):
        perm_map = np.arange(length)

        for t in indices_by_type:
            indices = np.copy(indices_by_type[t])
            np.random.shuffle(indices)  # Shuffle in-place
            perm_map[indices_by_type[t]] = indices

        permuted_indices.append(perm_map)

    return np.stack(permuted_indices, 0)

def sample_frac(atom_types, frac_coords, n_perm):
    """
    Sample a subset of permuted fractional coordinates.

    Args:
        group_indices (dict): {group_id -> array of indices}.
        frac_coords (np.array): Original fractional coordinates of shape (n_atoms, 3).
        n_perm (int): Number of permutations to sample.

    Returns:
        np.array: Sampled permuted fractional coordinates of shape (n_perm, n_atoms, 3),
                  padded with -1 if needed.
    """
    # Step 1: Compute the total number of valid permutations
    
    permuted_indices = generate_permutations(atom_types, frac_coords.shape[0], n_perm )
    n_actual = permuted_indices.shape[0]
    # Step 2: Apply the sampled permutations to frac_coords
    sampled_frac_coords = frac_coords[permuted_indices]

    # Step 3: Pad `sampled_frac_coords` with -1 if necessary
    if n_actual < n_perm:
        padding = -1 * np.ones((n_perm - n_actual, frac_coords.shape[0], frac_coords.shape[1]))
        sampled_frac_coords = np.vstack([sampled_frac_coords, padding])
    return sampled_frac_coords

def transform_data_to_cartesian(data):
    # Apply sine and cosine to each element in the data
    sin_vals = np.sin(2 * np.pi * data)
    cos_vals = np.cos(2 * np.pi * data)
    # Stack them into 2 columns (sin, cos for each dimension)
    return np.column_stack((sin_vals, cos_vals))

class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode, use_pos_index: ValueNode,
                 n_perm: int = 1, n_pos: int = 1, debug: bool = False, adaptive_permute: bool = False, atom_types_to_ids_path = None, include_identity = False,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.n_perm = n_perm
        self.n_pos = n_pos
        self.debug = debug 
        self.include_identity = include_identity
        self.preprocess(save_path, preprocess_workers, prop)
        self.preprocess_all_pos(atom_types_to_ids_path)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None
        self.beta_scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_schedulers(self, beta_scheduler, sigma_scheduler):
        import faiss
        self.beta_scheduler = beta_scheduler
        self.sigma_scheduler = sigma_scheduler
        self.indices = None
        if self.n_perm <= 1:
            return
        
        self.indices = []
        for data_dict in self.cached_data:
            permuted_frac_coords = data_dict['graph_arrays'][-1]
            permuted_frac_coords = permuted_frac_coords.reshape(permuted_frac_coords.shape[0], -1)
            transformed_data = transform_data_to_cartesian(permuted_frac_coords).astype(np.float32)
            index = faiss.IndexFlatL2(2 * permuted_frac_coords.shape[1])
            index.add(transformed_data)
            self.indices.append(index)

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def preprocess_all_pos(self, atom_types_to_ids_path):
        if atom_types_to_ids_path is not None:
            if os.path.exists(atom_types_to_ids_path):
                self.atom_types_to_ids = torch.load(atom_types_to_ids_path)
            else:
                atom_types_to_ids = defaultdict(list)
                for idx, d in enumerate(self.cached_data):
                    atom_types = tuple(sorted(self.cached_data[idx]['graph_arrays'][1]))
                    atom_types_to_ids[atom_types].append(idx)
                torch.save(atom_types_to_ids, atom_types_to_ids_path)
                self.atom_types_to_ids = atom_types_to_ids    
        else:
            self.atom_types_to_ids = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        if self.debug:
            index = 1
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        times, rand_x, input_frac_coords = None, None, None
        if self.n_perm > 0:
            (frac_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms, group_indices) = data_dict['graph_arrays']
            if self.beta_scheduler is not None and self.sigma_scheduler is not None and self.indices is not None:
                times = self.beta_scheduler.uniform_sample_t(1, device=self.device) 
                sigmas = self.sigma_scheduler.sigmas[times]
                frac_coords_ts = torch.Tensor(frac_coords)
                rand_x = torch.rand_like(frac_coords_ts).to(self.device)
                input_frac_coords = (frac_coords_ts + sigmas * rand_x) % 1.
                input_frac_coords_np = input_frac_coords.cpu().numpy().reshape(1, -1)
                transformed_input_frac_coords_np = transform_data_to_cartesian(input_frac_coords_np)
                _, closet_ids = self.indices[index].search(transformed_input_frac_coords_np, self.n_perm)
                permuted_frac_coords = permuted_frac_coords[closet_ids[0]]
                permuted_frac_coords[closet_ids[0] == -1] = np.full_like(permuted_frac_coords[0], -1)
                # print(closet_ids, permuted_frac_coords)

            else:
                if not self.include_identity:
                    permuted_frac_coords = sample_frac(atom_types, frac_coords, self.n_perm)
                else:
                    permuted_frac_coords = sample_frac(atom_types, frac_coords, self.n_perm - 1)
                    permuted_frac_coords = np.concatenate([permuted_frac_coords, frac_coords[np.newaxis, :, :]], axis=0)
            permuted_frac_coords = permuted_frac_coords.reshape(-1, permuted_frac_coords.shape[-1])
        else:
            (frac_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms) = data_dict['graph_arrays'][:7]
            permuted_frac_coords = None

        if self.n_pos > 1 and self.atom_types_to_ids is not None:
            atom_types_sorted = tuple(sorted(atom_types))
            all_ids = self.atom_types_to_ids[atom_types_sorted]
            
            if self.n_pos - 1 <= len(all_ids):
                sampled_ids = list(np.random.choice(all_ids, self.n_pos - 1, replace=False))
            else:
                sampled_ids = all_ids + list(np.random.choice(all_ids, self.n_pos - 1 - len(all_ids), replace=True))
            
            all_permuted_coords = [permuted_frac_coords if permuted_frac_coords is not None else frac_coords]
            for sampled_id in sampled_ids:
                
                if self.n_perm > 0:
                    new_frac_coords, new_atom_types = self.cached_data[sampled_id]['graph_arrays'][:2]
                    if not self.include_identity:
                        new_permuted_frac_coords = sample_frac(new_frac_coords, new_atom_types, self.n_perm)
                    else:
                        new_permuted_frac_coords = sample_frac(new_frac_coords, new_atom_types, self.n_perm - 1)
                        new_permuted_frac_coords = np.concatenate([new_permuted_frac_coords, new_frac_coords[np.newaxis, :, :]], axis=0)                    
                else:
                    new_permuted_frac_coords = self.cached_data[sampled_id]['graph_arrays'][0]
                all_permuted_coords.append(new_permuted_frac_coords)
            permuted_frac_coords = np.concatenate(all_permuted_coords, axis=0)
        # print("get_item", permuted_frac_coords.shape, frac_coords.shape)
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            permuted_frac_coords=torch.Tensor(permuted_frac_coords) if permuted_frac_coords is not None else None,
            helper_permuted_indices=torch.arange(len(atom_types)).repeat(self.n_perm) if permuted_frac_coords is not None else torch.arange(len(atom_types)),
            times=times,
            rand_x=rand_x, 
            input_frac_coords=input_frac_coords,
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        if self.use_perm:
            (frac_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms, permuted_frac_coords) = data_dict['graph_arrays']
            n_all_perm = permuted_frac_coords.shape[0]
            sampled_indices = np.random.choice(n_all_perm, self.n_perm, replace=(self.n_perm > n_all_perm))  # Without replacement
            permuted_frac_coords = permuted_frac_coords[sampled_indices]   
        else:
            (frac_coords, atom_types, lengths, angles, edge_indices,
            to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            permuted_frac_coords=torch.Tensor(permuted_frac_coords) if permuted_frac_coords is not None else None
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
