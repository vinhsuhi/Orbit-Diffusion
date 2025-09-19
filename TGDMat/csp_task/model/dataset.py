import torch
import os
import bz2
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from model.data_utils import (preprocess, add_scaled_lattice_prop)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Step 2: Apply the sampled permutations to frac_coords
    sampled_frac_coords = frac_coords[permuted_indices]

    return sampled_frac_coords

class MaterialDataset(Dataset):
    def __init__(self, data_dir, dataset, prompt_type, file, n_perm=0, include_identity=False):
        super().__init__()
        self.dataset = dataset
        self.path = os.path.join(data_dir, dataset, file + ".csv")
        self.df = pd.read_csv(self.path)
        self.prompt_type = prompt_type
        if self.dataset == 'perov_5':
            self.prop = ['heat_ref']
            all_attributes = ["pretty_formula", "elements", "heat_ref", "spacegroup", "system"]
        elif self.dataset == 'carbon_24':
            self.prop = ["energy_per_atom"]
            all_attributes = ["pretty_formula", "elements", "energy_per_atom", "spacegroup", "system"]
        elif self.dataset == 'mp_20':
            self.prop = ["formation_energy_per_atom", "band_gap", "e_above_hull"]
            all_attributes = ["pretty_formula", "elements", "formation_energy_per_atom", "band_gap", "e_above_hull",
                              "spacegroup", "system"]

        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.lattice_scale_method = 'scale_length'
        self.preprocess_workers = 30
        self.n_perm = n_perm
        self.include_identity = include_identity
        # print(self.path)
        # print(dataset)
        # print(file)
        cache_path = os.path.join(data_dir, dataset, file + ".pbz2")
        # print(cache_path)

        self.cached_data = None
        if os.path.exists(cache_path):
            try:
                print("Loading data...", end=" ", flush=True)
                with bz2.BZ2File(cache_path, "rb") as f:
                    self.cached_data = pickle.load(f)
            except Exception as e:
                print(e)
            else:
                print("done.")
        if self.cached_data is None:
            self.cached_data = preprocess(self.path,
                                      niggli=self.niggli,
                                      primitive=self.primitive,
                                      graph_method=self.graph_method,
                                      prop_list=self.prop,
                                      all_attributes = all_attributes,
                                      dataset = self.dataset)
            with bz2.BZ2File(cache_path, "wb") as f:
                pickle.dump(self.cached_data, f)



        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, edge_indices,to_jimages, num_atoms) = data_dict['graph_arrays']

        one_hot = np.zeros((len(atom_types), 100))
        for i in range(len(atom_types)):
            one_hot[i][atom_types[i]-1]=1

        if self.prompt_type=='long':
            text = data_dict['text_long']
        else:
            text = data_dict['text_short']
        if self.n_perm > 0:
            if not self.include_identity:
                permuted_frac_coords = sample_frac(atom_types, frac_coords, self.n_perm)
            else:
                permuted_frac_coords = sample_frac(atom_types, frac_coords, self.n_perm - 1)
                permuted_frac_coords = np.concatenate([permuted_frac_coords, frac_coords[np.newaxis, :, :]], axis=0)
                
            permuted_frac_coords = permuted_frac_coords.reshape(-1, permuted_frac_coords.shape[-1])
        else:
            permuted_frac_coords = None 
            
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            atom_types_one_hot=torch.Tensor(one_hot),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            text=text,
            permuted_frac_coords=torch.Tensor(permuted_frac_coords) if permuted_frac_coords is not None else None,
            helper_permuted_indices=torch.arange(len(atom_types)).repeat(self.n_perm) if permuted_frac_coords is not None else torch.arange(len(atom_types)),
        )
        return data


def main():
    return None


if __name__ == "__main__":
    main()