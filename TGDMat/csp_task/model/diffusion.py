import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_scatter import scatter, scatter_sum, scatter_mean

from tqdm import tqdm
# from model.decoder import MaTDecoder
from model.t2mnet import T2MNet
from model.diff_utils import BetaScheduler,SigmaScheduler
from model.diff_utils import d_log_p_wrapped_normal, d_log_p_wrapped_normal_stable
from model.data_utils import lattice_params_to_matrix_torch

MAX_ATOMIC_NUM=100
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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


class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=256,dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TGDiffusion(nn.Module):
    def __init__(self,timesteps, lr=0.00, include_identity=True, new_noise=False) -> None:
        super(TGDiffusion, self).__init__()
        self.decoder = T2MNet(hidden_dim = 512, max_atoms = 100, num_layers = 6, act_fn = 'silu', dis_emb = 'sin',
                              num_freqs = 128, edge_style = 'fc',max_neighbors= 20, cutoff = 7, ln = True, ip = True)
        self.text_projection = ProjectionHead(embedding_dim=768)
        self.beta_scheduler = BetaScheduler(timesteps=timesteps, scheduler_mode='cosine')
        self.sigma_scheduler = SigmaScheduler(timesteps=timesteps, sigma_begin=0.005, sigma_end=0.5)
        self.time_dim = 256
        self.shift_C = 2.0
        self.new_noise = new_noise
        self.include_identity = include_identity
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.6, patience=30, threshold=0.0001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.shift_C = 2.0
        if self.num_trans > 0:
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

        if self.z_coord is not None and type(self.z_coord) == float:
            temperature = torch.clamp(self.z_coord/sigmas, max=1.0).unsqueeze(-1)
        else:
            temperature = 1.0
        
        logits = flatten_repeated_log_p_xt_given_xo/temperature
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
    
    def forward(self, batch, baseline_aug=False, num_trans=0, weighted_coord_loss=False,z=None):
        self.num_trans=num_trans
        self.weighted_coord_loss=weighted_coord_loss
        self.z_coord = z
        text_emb = self.text_projection(batch.text)
        text_emb = torch.squeeze(text_emb)

        # Diffusion Forward
        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        if baseline_aug:
            frac_coords = batch.frac_coords
            random_shifts = torch.rand((batch_size, 3), device=self.device)
            frac_coords = (frac_coords + random_shifts[batch.batch]) % 1.0
            self.is_ours = False
        else:
            frac_coords = batch.frac_coords
            self.is_ours = batch.get("permuted_frac_coords", None) is not None or self.num_trans > 0

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        pred_l, pred_x = self.decoder(text_emb, time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        loss_coord, _, _ = self._loss_coord(sigmas_per_atom, rand_x, sigmas_norm_per_atom, batch, input_frac_coords, sigmas, pred_x)

        loss_lattice = F.mse_loss(pred_l, rand_l) # latice data prediction loss
        loss = loss_lattice +loss_coord

        return loss, loss_lattice, loss_coord

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):
        text_emb = self.text_projection(batch.text)
        text_emb = torch.squeeze(text_emb)

        batch_size = batch.num_graphs
        l_T, x_T = torch.randn([batch_size, 3, 3]).to(device), torch.rand([batch.num_nodes, 3]).to(device)
        time_start = self.beta_scheduler.timesteps
        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = device)

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

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # pred_l, pred_x, pred_type = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch)
            # pred_l, pred_x = self.decoder(text_emb, time_emb, batch.atom_types, x_t, l_t, batch)
            pred_l, pred_x = self.decoder(text_emb, time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            l_t_minus_05 = l_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # pred_l, pred_x, pred_type = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch)
            # pred_l, pred_x = self.decoder(text_emb, time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05,batch)
            pred_l, pred_x = self.decoder(text_emb, time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l


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

    