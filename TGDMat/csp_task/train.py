import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from time import time
from config import config
from model.diffusion import TGDiffusion
from torch_geometric.data import DataLoader
from model.dataset import MaterialDataset
from utils import save_model
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_size(model):
    # Get total number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Each parameter is typically 4 bytes (float32)
    model_size_in_bytes = num_params * 4  # 4 bytes for float32

    # Convert to MB
    model_size_in_MB = model_size_in_bytes / (1024 ** 2)

    return model_size_in_MB, num_params

def train_epoch(dataloader, model, additional_train_args):
    iters = len(dataloader)
    diff_losses, coord_losses, lattice_losses = np.empty(iters), np.empty(iters), np.empty(iters)

    for i, batch in enumerate(dataloader):
        batch=batch.to(device)
        loss, loss_lattice, loss_coord  = model(batch, **additional_train_args)
        model.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        model.optim.step()

        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        lattice_losses[i] = loss_lattice.item()

    return diff_losses.mean(), coord_losses.mean(), lattice_losses.mean()

def validate(dataloader, model):
    iters = len(dataloader)
    diff_losses, coord_losses, lattice_losses = np.empty(iters), np.empty(iters), np.empty(iters)

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        loss, loss_lattice, loss_coord  = model(batch)
        diff_losses[i] = loss.item()
        coord_losses[i] = loss_coord.item()
        lattice_losses[i] = loss_lattice.item()

    return diff_losses.mean(), coord_losses.mean(), lattice_losses.mean()



def train(train_dataloader, model, epochs, additional_train_args, args):
    basedir = 'out'
    exp_name = args.exp_name

    wandb.init(
        project="tgdmat",
        name=exp_name,
        config=vars(args)
    )
    
    path = os.path.join(basedir, exp_name, config.file_name_model)


    os.makedirs(path, exist_ok=True)

    for epoch in tqdm(range(args.resume_epoch, epochs)):
        t0 = time()
        loss, coord_loss, lattice_loss = train_epoch(train_dataloader, model, additional_train_args)
        model.scheduler.step(loss)
        wandb.log({
            "epoch": epoch,
            "train/diff_loss": loss,
            "train/coord_loss": coord_loss,
            "train/lattice_loss": lattice_loss,
            "time_per_epoch": time() - t0
        })
        
    save_model(model, f"{path}_final.pt")
    print("All saved at ", path)
    return model,path



def main(args):
    torch.manual_seed(config.SEED)

    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"

    include_identity = True
        
    model = TGDiffusion(args.timesteps, lr=args.lr, include_identity=include_identity, new_noise=args.new_noise).to(device)
    if args.pretrained:
        chkpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(chkpt["model"])
        if args.resume_epoch:
            model.optim.load_state_dict(chkpt["model_opt"])

    model_size_in_MB, num_params = get_model_size(model)
    print("Total Params: ", num_params)
    print("Model Size (in MB): ",round(model_size_in_MB,2) )
    print("Prompt Type ",args.prompt_type)
    root_path = "../data_text/"

    train_dataset = MaterialDataset(root_path, args.dataset, args.prompt_type, config.train_data, n_perm=args.n_perm, include_identity=include_identity)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    additional_train_args = {
        'num_trans': args.num_trans,
    }
    train(train_dataloader, model, args.epochs, additional_train_args, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--prompt_type', type=str, default='long')  # long or short
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--n_perm', type=int, default=0)
    parser.add_argument('--num_trans', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--new_noise', action='store_true')
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--resume_epoch', type=int, default=0)
    args = parser.parse_args()
    
    import random
    import numpy as np
    import torch

    # Set random seed for Python
    random.seed(args.seed)

    # Set random seed for NumPy
    np.random.seed(args.seed)

    # Set random seed for PyTorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
