import argparse
import datetime
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch
import wandb

# from lightning import seed_everything
from loguru import logger as log
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from etflow.commons import get_base_data_dir, load_pkl, save_pkl
from etflow.models import BaseFlow
from etflow.utils import instantiate_dataset, instantiate_model, read_yaml
import pickle
import datamol as dm

torch.set_float32_matmul_precision("high")


def get_datatime():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def cuda_available():
    return torch.cuda.is_available()


def main(
    config: str,
    checkpoint_path: str,
    dataframe_path: str,
    indices: np.ndarray,
    counts: np.ndarray,
):
    if cuda_available():
        log.info("CUDA is available. Using GPU for sampling.")
        device = torch.device("cuda")
    else:
        log.warning("CUDA is not available. Using CPU for sampling.")
        device = torch.device("cpu")

    # instantiate datamodule and model
    dataset = instantiate_dataset(
        config["datamodule_args"]["dataset"], config["datamodule_args"]["dataset_args"], train_mode=config["train_mode"]
    )
    
    model = instantiate_model(config["model"], config["model_args"], config["train_mode"], config["remove_rotation"])

    # load model weights
    log.info(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    # move to device
    model: BaseFlow = model.to(device)
    model.eval()

    # max batch size
    max_batch_size = config["batch_size"]

    # load indices
    # IDXS = [72139, 72140, 72141, 72142, 72143, 72144, 72145, 72146]
    IDXS = np.load('/data/work/ac141281/IEF_processed_data/QM9/train_indices_1sample.npy')
    IDX = IDXS[0]
    data = dataset[IDX]
    # get data for batch_size
    smiles = data.smiles
    log.info(f"Generating conformers for molecule: {smiles}")
    # calculate number of samples to generated
    test_overfit = True
    if test_overfit:
        count = len(IDXS)
        num_samples = 10 * count
        all_pos = np.array([dataset[i].pos.numpy() for i in IDXS])
    else:
        smile_to_idx = pickle.load(open("/data/work/ac141281/IEF/smile_to_pos/smiles_to_idx.pkl", "rb"))
        smile_idx = smile_to_idx[smiles]
        all_pos = pickle.load(open(f"/data/work/ac141281/IEF/smile_to_pos/{smile_idx}.pkl", "rb"))
        if config["remove_hs"]:
            mol_with_hs = dm.to_mol(smiles, remove_hs=False, ordered=True)
            not_h_indices = [i for i, atom in enumerate(mol_with_hs.GetAtoms()) if atom.GetSymbol() != 'H']
            all_pos = all_pos[:, not_h_indices, :]
                
        count = all_pos.shape[0]
        num_samples = 10 * count
    # we would want (num_samples, num_nodes, 3)
    generated_positions = []
    times = []

    for batch_start in range(0, num_samples, max_batch_size):
        # get batch_size
        batch_size = min(max_batch_size, num_samples - batch_start)

        # batch the data
        batched_data = Batch.from_data_list([data] * batch_size)

        # get one_hot, edge_index, batch
        (
            z,
            edge_index,
            batch,
            node_attr,
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        ) = (
            batched_data["atomic_numbers"].to(device),
            batched_data["edge_index"].to(device),
            batched_data["batch"].to(device),
            batched_data["node_attr"].to(device),
            batched_data["chiral_index"].to(device),
            batched_data["chiral_nbr_index"].to(device),
            batched_data["chiral_tag"].to(device),
        )

        # get time-estimate
        start = time.time()
        with torch.no_grad():
            # generate samples
            pos = model.sample(
                z,
                edge_index,
                batch,
                node_attr=node_attr,
                n_timesteps=config["nsteps"],
                chiral_index=chiral_index,
                chiral_nbr_index=chiral_nbr_index,
                chiral_tag=chiral_tag,
                s_churn=config["churn"],
                t_min=config["t_min"],
                t_max=config["t_max"],
                std=config["std"],
                sampler_type=config["sample_type"],
            )
        end = time.time()
        times.append((end - start) / batch_size)  # store time per conformer

        # reshape to (num_samples, num_atoms, 3) using batch
        pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

        # append to generated_positions
        generated_positions.append(pos)
    
    # concatenate generated_positions: (num_samples, num_atoms, 3)
    generated_positions = np.concatenate(generated_positions, axis=0)

    # save to file
    path = osp.join(output_dir, f"{IDX}.pkl")
    log.info(f"Saving generated positions to file for smiles {smiles} at {path}")
    save_pkl(path, generated_positions)
    save_pkl(os.path.join(output_dir, "times.pkl"), times)

    data = Data(
            smiles=smiles, pos_ref=all_pos, rdmol=dataset[IDX].mol, pos_gen=generated_positions, remove_hs=config["remove_hs"]
        )
    
    save_pkl(os.path.join(output_dir, "generated_files.pkl"), [data])
    print(os.path.join(output_dir, "generated_files.pkl"))
    


if __name__ == "__main__":
    # argparse checkpoint path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-k", type=str, required=True)
    parser.add_argument(
        "--count_indices", "-i", type=str, required=False, default="scaffold_qm9_count_indices.npy"
    )
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="logs/")
    parser.add_argument(
        "--dataset_type", "-t", type=str, required=False, default="drugs"
    )
    parser.add_argument("--sampler_type", "-s", type=str, required=False, default="ode")
    parser.add_argument("--batch_size", "-b", type=int, required=False, default=32)
    parser.add_argument("--nsteps", "-n", type=int, required=False, default=50)
    parser.add_argument("--churn", "-ch", type=float, required=False, default=1.0)
    parser.add_argument("--t_min", "-mn", type=float, required=False, default=0.0001)
    parser.add_argument("--t_max", "-mx", type=float, required=False, default=0.9999)
    parser.add_argument("--std", "-st", type=float, required=False, default=1.0)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--train_mode", type=str, default="baseline")
    parser.add_argument("--test_name", type=str, default="")
    parser.add_argument("--remove_rotation", action="store_true")
    parser.add_argument("--remove_auto", action="store_true")
    parser.add_argument("--remove_hs", action="store_true")

    args = parser.parse_args()

    # debug mode
    debug = args.debug
    log.info(f"Debug mode: {debug}")

    # base data
    DATA_DIR = get_base_data_dir()

    # set dataframe path
    df_path = osp.join(DATA_DIR, "processed", "geom.csv")
    assert osp.exists(df_path), f"Dataframe path {df_path} not found"

    # read config
    assert osp.exists(args.config), "Config path does not exist."
    log.info(f"Loading config from: {args.config}")
    config = read_yaml(args.config)
    config["train_mode"] = args.train_mode
    config["test_name"] = args.test_name
    
    config["remove_rotation"] = args.remove_rotation
    config["remove_auto"] = args.remove_auto
    config["remove_hs"] = args.remove_hs
    
    # task_name = config.get("task_name", "default")

    # task_name = config.get("train_mode", "default")

    task_name = args.test_name

    # start wandb
    if not debug:
        # log experiment info
        log_dict = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "dataset_type": args.dataset_type,
            "sampler_type": args.sampler_type,
            "debug": debug,
        }

    # get checkpoint path
    checkpoint_path = args.checkpoint
    assert osp.exists(checkpoint_path), "Checkpoint path does not exist."

    # setup output directory for storing samples
    output_dir = osp.join(
        args.output_dir,
        f"samples/{task_name}/flow_nsteps_{args.nsteps}",
    )
    if not debug:
        os.makedirs(output_dir, exist_ok=True)

    # load count indices path, indices for what smiles to use
    count_indices_path = osp.join(
        DATA_DIR, args.dataset_type.upper(), args.count_indices
    )
    assert osp.exists(
        count_indices_path
    ), f"Count indices path: {count_indices_path} does not exist."
    log.info(f"Loading count indices from: {count_indices_path}")
    indices, counts = np.load(count_indices_path)
    log.info(f"Will be generating samples for {len(indices)} counts.")

    # update config
    config.update(
        {
            "nsteps": args.nsteps,
            "churn": args.churn,
            "t_min": args.t_min,
            "t_max": args.t_max,
            "batch_size": args.batch_size,
            "subset_type": args.dataset_type,
            "sample_type": args.sampler_type,
            "std": args.std,
        }
    )
    # set dataframe path
    dataframe_path = osp.join(DATA_DIR, "processed", "geom.csv")

    main(config, checkpoint_path, dataframe_path, indices, counts)
