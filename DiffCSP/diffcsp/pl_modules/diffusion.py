import math, copy
import time, os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

import hydra
import pytorch_lightning as pl
from torch_scatter import scatter, scatter_sum
from tqdm import tqdm

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal_stable

MAX_ATOMIC_NUM=100

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model # type: ignore

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial" # type: ignore
        )
        if not self.hparams.optim.use_lr_scheduler: # type: ignore
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt # type: ignore
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False) # type: ignore
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler) # type: ignore
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler) # type: ignore
        self.time_dim = self.hparams.time_dim # type: ignore
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim) # type: ignore
        self.keep_lattice = self.hparams.cost_lattice < 1e-5 # type: ignore
        self.keep_coords = self.hparams.cost_coord < 1e-5 # type: ignore
        
        self.num_trans = kwargs.get("num_trans", 0)
        self.include_identity = kwargs.get("include_identity", False)
        self.new_noise = kwargs.get("new_noise", False)
        self.shift_C = float(kwargs.get("shift_C", 2.0))
        
    def compute_log_p(self, x, sigmas_per_atom):
        # x: 1, n_trans, n_atoms, 3
        k = torch.arange(-10, 11).to(self.device).view(-1,1,1,1)
        inv_sigma = 1.0 / (sigmas_per_atom**2)
        log_p_xt_given_xo = torch.logsumexp(-0.5 * ((x + k)**2 * inv_sigma), dim=0)
        log_p_xt_given_xo = log_p_xt_given_xo.sum(dim=-1)
        return log_p_xt_given_xo
        
    def compute_log_p_and_d_log_p(self, x, sigmas_per_atom, batch):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        log_p_xt_given_xo = self.compute_log_p(x, sigmas_per_atom)
        log_p_xt_given_xo = scatter_sum(log_p_xt_given_xo, batch, dim=1)
        permuted_tar_x = d_log_p_wrapped_normal_stable(x, sigmas_per_atom) 
        return log_p_xt_given_xo, permuted_tar_x
    
    def _transition_frac_coords(self, permuted_frac_coords, batch_size, num_permutations, sigmas, repeated_batch):
        if self.num_trans > 0:
            # stime = time.time() 
            if not self.new_noise:
                random_shifts = torch.rand((self.num_trans, batch_size * num_permutations, 3), device=self.device)
                if self.include_identity:
                    random_shifts[0] *= 0
                log_density = 0
            else:                                                                 
                random_shifts = torch.randn((self.num_trans, batch_size * num_permutations, 3), device=self.device)
                new_sigmas = self.shift_C * sigmas.repeat_interleave(num_permutations)[:,None].unsqueeze(0)
                random_shifts = random_shifts * new_sigmas
                if self.include_identity:
                    random_shifts[0] *= 0
                    
                log_density = self.compute_log_p(random_shifts, new_sigmas)

            expanded_shifts = random_shifts[:, repeated_batch]
        
            # get all permuted version of frac_coords
            all_permuted_frac_coords = (permuted_frac_coords.unsqueeze(0) + expanded_shifts + 1) % 1.
        else:
            all_permuted_frac_coords = permuted_frac_coords.unsqueeze(0)
            log_density = 0
            
        return all_permuted_frac_coords, log_density
    
    def _compute_target_score(self, frac_coords_t, sigmas, batch):
        batch_size = batch.num_graphs
        if batch.get("permuted_frac_coords", None) is not None:
            permuted_frac_coords = batch.permuted_frac_coords
        else:
            permuted_frac_coords = batch.frac_coords
            
        num_permutations = permuted_frac_coords.shape[0] // batch.frac_coords.shape[0]
            
        # duplicate batch.batch to match permuted_frac_coords
        repeated_batch = torch.arange(batch.num_atoms.shape[0] * num_permutations).to(self.device).repeat_interleave(batch.num_atoms.repeat_interleave(num_permutations))
        batch_size = batch.num_atoms.shape[0]
        
        # duplicate sigmas stuff to match all_permuted_frac_coords
        if type(sigmas) is torch.Tensor:
            repeated_sigmas_per_atom = sigmas.repeat_interleave(num_permutations*batch.num_atoms)[:,None]
        else:
            repeated_sigmas_per_atom = sigmas
            
        all_permuted_frac_coords, log_density = self._transition_frac_coords(permuted_frac_coords, batch_size, num_permutations, sigmas, repeated_batch)

        # duplicate frac_coords_t to match all_permuted_frac_coords
        
        sample_offset = torch.cumsum(batch.num_atoms, dim=0)
        sample_offset_full = torch.cat((torch.tensor([0]).to(self.device), sample_offset))
        repeated_sample_offset_full = torch.cat([torch.tile(torch.arange(sample_offset_full[i], sample_offset_full[i+1]).to(self.device), (num_permutations,)) for i in range(len(sample_offset_full)-1)]) # type: ignore
        repeated_frac_coords_t = frac_coords_t[repeated_sample_offset_full]

        repeated_log_p_xt_given_xo, repeated_permuted_tar_x = self.compute_log_p_and_d_log_p(repeated_frac_coords_t - all_permuted_frac_coords, repeated_sigmas_per_atom, repeated_batch)
        repeated_log_p_xt_given_xo -= log_density
        
        flatten_repeated_log_p_xt_given_xo = torch.stack([t.flatten() for t in repeated_log_p_xt_given_xo.chunk(batch_size,-1)])
        
        logits = flatten_repeated_log_p_xt_given_xo
        flatten_weights = torch.nn.functional.softmax(logits, dim=-1)
            
        weights = torch.cat([w.view(-1, num_permutations) for w in flatten_weights],-1)
        repeated_tar_x = (weights[:,repeated_batch].unsqueeze(-1) * repeated_permuted_tar_x).sum(0)
        tar_score_x = scatter(repeated_tar_x, repeated_sample_offset_full, dim=0, reduce="sum")

        return tar_score_x
    
    def _loss_coord(self, sigmas_per_atom, rand_x, sigmas_norm_per_atom, batch, input_frac_coords, sigmas, pred_score_x ):         
        if self.is_ours:
            tar_score_x = self._compute_target_score(input_frac_coords, sigmas, batch) / torch.sqrt(sigmas_norm_per_atom) 
            loss_coord = F.mse_loss(pred_score_x, tar_score_x)
            baseline_loss_coord = 0
            ours_loss_coord = loss_coord
        else:
            tar_score_x = d_log_p_wrapped_normal_stable(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
            loss_coord = F.mse_loss(pred_score_x, tar_score_x)
            baseline_loss_coord = loss_coord
            ours_loss_coord = 0

        return loss_coord, baseline_loss_coord, ours_loss_coord 
          
    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        frac_coords = batch.frac_coords
        self.is_ours = batch.get("permuted_frac_coords", None) is not None or self.num_trans > 0

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_noise_l, pred_score_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        loss_coord, baseline_loss_coord, ours_loss_coord = self._loss_coord(sigmas_per_atom, rand_x, sigmas_norm_per_atom, batch, input_frac_coords, sigmas, pred_score_x)
            
        loss_lattice = F.mse_loss(pred_noise_l, rand_l) # latice data prediction loss

        loss = (
            self.hparams.cost_lattice * loss_lattice + # type: ignore
            self.hparams.cost_coord * loss_coord) # type: ignore

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'baseline_loss_coord' : baseline_loss_coord,
            'ours_loss_coord' : ours_loss_coord,
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            # step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='train')

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs
        )

        if loss.isnan():
            return None # type: ignore

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_{k}'.strip("_"): v for k, v in output_dict.items()
        }

        return log_dict, loss
