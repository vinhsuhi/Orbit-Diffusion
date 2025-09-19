from config import config
import torch
from data.dataset import MaterialDataset
from torch_geometric.loader import DataLoader
import numpy as np
import random
from matplotlib import pyplot as plt

def main():
    # train_dataset = MaterialDataset(f"data/{config.dataset}", config.data, config.device)
    dataset = MaterialDataset(f"data/{config.dataset}", config.eval_data, config.device)
    test_loader = DataLoader(dataset, batch_size=config.batch_size)
    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"

    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    atom_type_distribution = [0]*100
    for i in range(len(dataset)):
        num_atoms, atom_type, frac_coord, L, one_hot, n_mask, d = dataset[i]
        atom_type = atom_type.tolist()
        atom_type_flatten = [item[0] for item in atom_type]
        atom_type_dist = [0]*100
        for k in set(atom_type_flatten):
            atom_type_dist[k-1] = atom_type_flatten.count(k)/len(atom_type_flatten)
        for j in range(len(atom_type_distribution)):
            atom_type_distribution[j]= atom_type_distribution[j]+atom_type_dist[j]
        # print(dataset[i])
        # exit()

    data = [x / len(atom_type_distribution) for x in atom_type_distribution]

    courses = range(len(data))
    values = data

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color='maroon', width=0.4)
    # plt.xticks(range(len(data)))

    plt.xlabel("Courses offered")
    plt.ylabel("No. of students enrolled")
    plt.title("Students enrolled in different courses")
    plt.show()


    # for idx, batch in enumerate(test_loader):
    #     print(f'batch {idx} in {len(test_loader)}')
    #     batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
    #     batch_lengths, batch_angles = [], []
    #
    #     num_atoms_gt, A, frac_coord_gt, L, one_hot, n_mask, d = batch
    #     max_atoms = torch.max(num_atoms_gt)
    #     length_gt = L[:, :3].clone()
    #     angle_gt = L[:, 3:].clone()
    #
    #     # assert frac_coords.size(0) == atom_types.size(0)
    #     # assert frac_coords.size(0) == atom_types.size(0)
    #     # assert frac_coords.size(0) == atom_types.size(0)
    #     # assert frac_coords.size(0) == atom_types.size(0)
    #
    #     # collect sampled crystals in this batch.
    #     frac_coords.append(frac_coord_gt)
    #     num_atoms.append(num_atoms_gt)
    #     atom_types.append(A)
    #     lengths.append(length_gt)
    #     angles.append(angle_gt)
    #
    #     # frac_coords.append(torch.stack(batch_frac_coords, dim=0))
    #     # num_atoms.append(torch.stack(batch_num_atoms, dim=0))
    #     # atom_types.append(torch.stack(batch_atom_types, dim=0))
    #     # lengths.append(torch.stack(batch_lengths, dim=0))
    #     # angles.append(torch.stack(batch_angles, dim=0))
    #     # input_data_list = input_data_list + d.to_data_list()
    #
    # frac_coords = torch.concat(frac_coords, dim=0)
    # num_atoms = torch.concat(num_atoms, dim=0)
    # atom_types = torch.concat(atom_types, dim=0)
    # lengths = torch.concat(lengths, dim=0)
    # angles = torch.concat(angles, dim=0)






if __name__ == '__main__':
    main()