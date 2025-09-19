import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import config
from torch_geometric.data import Batch
from model.dataset import MaterialDataset
from torch_geometric.data import DataLoader
from model.diffusion import TGDiffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

recommand_step_lr = {
    'csp':{
        "perov_5": 5e-7,
        "carbon_24": 5e-6,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'csp_multi':{
        "perov_5": 5e-7,
        "carbon_24": 5e-7,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'gen':{
        "perov_5": 1e-6,
        "carbon_24": 1e-5,
        "mp_20": 5e-6
    },
}

def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) / (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles



def reconstructon(loader, model, num_evals, step_lr = 1e-5):
    """
    reconstruct the crystals in <loader>.
    """
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            print(f'batch {idx+1} / {len(loader)}, sample {eval_idx+1} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr=step_lr)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch)

def generation(loader, model, step_lr):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr=step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)


def main(args):
    print("Arguments:", args)
    model_path = Path(args.model_path)
    root_path = "../data_text/"

    test_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.test_data)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"

    chkpt_name = args.chkpt_path

    model = TGDiffusion(args.timesteps).to(device)

    chkpt = torch.load(chkpt_name, map_location=device)
    model.load_state_dict(chkpt["model"])

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi']['perov_5']
    if torch.cuda.is_available():
        model.to('cuda')

    if 'csp' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords,  atom_types,_, lengths, angles,num_atoms, input_data_batch) = reconstructon(test_dataloader,model, args.num_evals, step_lr)
        print('Reconstruction Time :', time.time() - start_time)
        recon_out_name = f'eval_recon_{args.label}.pt'
        PATH = model_path / recon_out_name
        print('Saving Pt File..')
        print(PATH)
        torch.save({'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'time': time.time() - start_time
        }, PATH)
        print('Saving Pt File..Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--chkpt_path', required=True, type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--prompt_type', type=str, default='long')  # long or short
    parser.add_argument('--label', type=str, default='')  # long or short
    args = parser.parse_args()
    main(args)
    # main('gen/',"30112023","001402",'recon',8,4)