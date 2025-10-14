from platform import node
from typing import Any, Dict, List, Optional, TypeVar, Union

from matplotlib.pyplot import sca
import numpy as np
from pytz import utc
import torch
from pytorch_lightning import seed_everything
from torch import Tensor
from torch_geometric.data import Batch

from etflow.commons.configs import CONFIG_DICT
from etflow.commons.covmat import set_multiple_rdmol_positions
from etflow.commons.featurization import MoleculeFeaturizer, get_mol_from_smiles
from etflow.commons.utils import signed_volume
from etflow.models.base import BaseModel
from etflow.models.loss import batchwise_l2_loss
from etflow.models.utils import (
    HarmonicSampler,
    center_of_mass,
    extend_bond_index,
    rmsd_align,
    unsqueeze_like,
)
from etflow.networks.torchmd_net import TorchMDDynamics

from etflow.commons.align_utils import random_rotation_matrices
from torch_scatter import scatter

__all__ = ["BaseFlow"]

Config = TypeVar("Config", str, Dict[str, Any])

#### new noise #####
import torch
import numpy as np

def compute_rotation_matrices_rodrigues(mu, target_vectors):
    """
    Compute Nxkx3x3 rotation matrices using Rodrigues' rotation formula.

    Args:
        mu (torch.Tensor): Source vector of shape (3,), typically [0, 0, 1].
        target_vectors (torch.Tensor): Target vectors of shape (N, k, 3).

    Returns:
        torch.Tensor: Rotation matrices of shape (N, k, 3, 3).
    """
    N1, N2, _ = target_vectors.shape
    mu = mu.view(1, 1, 3).expand(N1, N2, 3)

    # Calculate cross products (Nxkx3)
    k = torch.cross(mu, target_vectors, dim=-1)

    # Calculate dot products for cos(theta) (Nxkx1)
    cos_theta = (mu * target_vectors).sum(dim=-1, keepdim=True)

    # Calculate sine using the norm of cross product
    sin_theta = torch.norm(k, dim=-1, keepdim=True)

    # Normalize k to prevent division by zero (numerical stability)
    k = k / (sin_theta + 1e-8)

    # Create the skew-symmetric matrix K (Nxkx3x3)
    K = torch.zeros((N1, N2, 3, 3), device=target_vectors.device)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] =  k[..., 0]

    # Identity matrix (Nxkx3x3)
    I = torch.eye(3, device=target_vectors.device).view(1, 1, 3, 3).expand(N1, N2, 3, 3)

    # Ensure sin_theta and cos_theta are broadcastable
    sin_theta = sin_theta[..., None]
    cos_theta = cos_theta[..., None]

    # Compute rotation matrices using Rodrigues' formula
    R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)

    return R

def compute_rotation_matrices(v1, v2):
    """
    Compute the rotation matrix that rotates v1 to v2 using the Rodrigues' rotation formula.
    
    v1: A tensor of shape (1, 3), the initial vector.
    v2: A tensor of shape (n, 3), the target vectors.
    
    Returns:
        A tensor of shape (n, 3, 3), where each 3x3 matrix is the rotation matrix
        that rotates v1 to the corresponding row in v2.
    """
    # Normalize the vectors
    v1 = v1 / v1.norm(dim=1, keepdim=True)
    v2 = v2 / v2.norm(dim=1, keepdim=True)

    # Expand v1 to have the same shape as v2 (n, 3)
    v1_expanded = v1.expand(v2.size(0), -1)

    # Compute the angle theta between v1 and v2
    dot_prod = torch.sum(v1_expanded * v2, dim=1)  # Dot product for each pair of vectors
    theta = torch.acos(dot_prod)  # Angle between the vectors

    # Compute the cross product
    cross_prod = torch.cross(v1_expanded, v2, dim=1)  # Cross product along the second dimension (3)

    # Create the skew-symmetric cross product matrix K
    K = torch.zeros((v2.size(0), 3, 3), device=v1.device)
    K[:, 0, 1] = -cross_prod[:, 2]
    K[:, 0, 2] = cross_prod[:, 1]
    K[:, 1, 0] = cross_prod[:, 2]
    K[:, 1, 2] = -cross_prod[:, 0]
    K[:, 2, 0] = -cross_prod[:, 1]
    K[:, 2, 1] = cross_prod[:, 0]

    # Create the identity matrix I
    identity = torch.eye(3, device=v1.device).unsqueeze(0).expand(v2.size(0), -1, -1)

    # Compute the rotation matrix using Rodrigues' formula
    rotation_matrix = identity + torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * K + \
                      (1 - torch.cos(theta)).unsqueeze(-1).unsqueeze(-1) * torch.bmm(K, K)

    return rotation_matrix

class BaseFlow(BaseModel):
    """LightningModule for Flow Matching"""

    __prior_types__ = ["gaussian", "harmonic"]
    __interpolation_types__ = ["linear", "gvp", "gvp_w_sigma", "gvp_squared"]
    __path_types__ = ["standard", "cond_ot_path"]

    def __init__(
        self,
        # flow matching network args
        network_type: str = "TorchMDDynamics",
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = False,
        clip_during_norm: bool = False,
        max_num_neighbors: int = 32,
        so3_equivariant: bool = False,
        # flow matching args
        sigma: float = 0.1,
        interpolation_type: str = "linear",
        prior_type: str = "gaussian",
        sample_time_dist: Union[str, float] = "uniform",
        harmonic_alpha: float = 1.0,
        path_type: str = "standard",
        parity_switch: Optional[str] = None,
        # grad norm max value
        grad_norm_max_val: float = 100.0,
        # make edge_type one_hot
        edge_one_hot: bool = False,
        edge_one_hot_types: int = 5,
        # optimizer
        optimizer_type: str = "Adam",
        lr: float = 1e-3,
        beta1: float = 0.95,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        # lr scheduler args
        lr_scheduler_type: Optional[str] = "plateau",
        factor: float = 0.6,
        patience: int = 10,
        first_cycle_steps: int = 1000,
        cycle_mult: float = 1.0,
        max_lr: float = 0.0001,
        min_lr: float = 1.0e-08,
        warmup_steps: int = 10000,
        gamma: float = 0.75,
        last_epoch: int = -1,
        lr_scheduler_monitor: str = "val/loss",
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
        train_mode: str = "baseline",
        num_rotations: int = 1,
        z: float = 1.0,
        predict_mode: str = 'v',
    ):
        super().__init__(
            optimizer_type=optimizer_type,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            grad_norm_max_val=grad_norm_max_val,
            lr_scheduler_type=lr_scheduler_type,
            factor=factor,
            patience=patience,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            gamma=gamma,
            last_epoch=last_epoch,
            lr_scheduler_monitor=lr_scheduler_monitor,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_frequency=lr_scheduler_frequency,
        )

        # setup network
        if network_type == "TorchMDDynamics":
            self.network = TorchMDDynamics(
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_rbf=num_rbf,
                rbf_type=rbf_type,
                trainable_rbf=trainable_rbf,
                activation=activation,
                neighbor_embedding=neighbor_embedding,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=max_z,
                node_attr_dim=node_attr_dim,
                edge_attr_dim=edge_attr_dim,
                attn_activation=attn_activation,
                num_heads=num_heads,
                distance_influence=distance_influence,
                reduce_op=reduce_op,
                qk_norm=qk_norm,
                output_layer_norm=output_layer_norm,
                clip_during_norm=clip_during_norm,
                so3_equivariant=so3_equivariant,
            )
        else:
            raise NotImplementedError(
                f"Network type {network_type} not implemented for BaseFlow"
            )

        self.sigma = sigma
        self.cutoff = cutoff_upper
        self.parity_switch = parity_switch
        self.prior_type = prior_type
        self.interpolation_type = interpolation_type
        self.sample_time_dist = sample_time_dist
        self.edge_one_hot = edge_one_hot
        self.edge_one_hot_types = edge_one_hot_types
        self.max_num_neighbors = max_num_neighbors
        self.path_type = path_type

        self.train_mode = train_mode
        self.num_rotations = num_rotations
        self.predict_mode = predict_mode
        assert predict_mode in ['v', 'x']
        self.z = z 

        # TODO:remove it later
        # assert self.num_rotations == 1, "we don't use rotation for now"
        assert self.interpolation_type == "linear", "We only use linear interpolation"

        if parity_switch is not None:
            assert (
                parity_switch == "post_hoc"
            ), f"Parity switch {parity_switch} not implemented"

        assert (
            self.prior_type in self.__prior_types__
        ), f"""\nPrior type {prior_type} not available.
            This is the list of implemented prior types {self.__prior_types__}.\n"""

        assert (
            self.interpolation_type in self.__interpolation_types__
        ), f"""\Interpolation type {interpolation_type} not available.
            This is the list of implemented interpolations {self.__interpolation_types__}.\n"""

        assert (
            self.path_type in self.__path_types__
        ), f"""\Interpolation type {path_type} not available.
            This is the list of implemented interpolations {self.__path_types__}.\n"""

        if prior_type == "harmonic":
            self.harmonic_sampler = HarmonicSampler(alpha=harmonic_alpha)

    @classmethod
    def from_config(cls, cfg: Config):
        import yaml

        from etflow.utils import instantiate_model

        if isinstance(cfg, str):
            cfg = yaml.safe_load(open(cfg))
        if isinstance(cfg, dict):
            return instantiate_model(cfg["model"], cfg["model_args"])
        else:
            raise ValueError("cfg should be a dictionary or a path to a yaml file")

    @classmethod
    def from_default(
        cls, model: str = "drugs-o3", device: str = "cuda", cache: Optional[str] = None
    ):
        model = model.lower()
        if model not in CONFIG_DICT:
            raise ValueError(
                f"Model config {model} not found. Available checkpoints are {CONFIG_DICT.keys()}"
            )
        else:
            config = CONFIG_DICT.get(model, None)()
            print(f"Loading {model} from config")
            config.checkpoint_config.set_cache(cache)
            checkpoint_path = config.checkpoint_config.fetch_checkpoint().local_path

        found_device = get_device()
        if device != found_device and device != "cpu":
            print(f"Device {device} not found. Using {found_device} instead")
            device = found_device

        etflow_model = cls.from_config(config.model_dict())
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                # Standard Lightning checkpoint
                etflow_model.load_state_dict(checkpoint["state_dict"])
            else:
                # Plain state dict
                etflow_model.load_state_dict(checkpoint)
        etflow_model.eval()
        return etflow_model

    def alpha_t(self, t):
        if self.interpolation_type == "linear":
            return t
        elif self.interpolation_type == "gvp":
            return torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return torch.sqrt(1 - self.sigma_t(t) ** 2) * torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_squared":
            return torch.sin(0.5 * torch.pi * t) ** 2

    def beta_t(self, t):
        if self.interpolation_type == "linear":
            return 1 - t
        elif self.interpolation_type == "gvp":
            return torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return torch.sqrt(1 - self.sigma_t(t) ** 2) * torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_squared":
            return torch.cos(0.5 * torch.pi * t) ** 2

    def alpha_dot_t(self, t):
        if self.interpolation_type == "linear":
            return 1
        elif self.interpolation_type == "gvp":
            return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return -self.sigma_dot_t(t) / torch.sqrt(
                1 - self.sigma_t(t) ** 2
            ) * torch.sin(0.5 * torch.pi * t) + 0.5 * torch.pi * torch.cos(
                0.5 * torch.pi * t
            )
        elif self.interpolation_type == "gvp_squared":
            return (
                torch.pi * torch.sin(0.5 * torch.pi * t) * torch.cos(0.5 * torch.pi * t)
            )

    def beta_dot_t(self, t):
        if self.interpolation_type == "linear":
            return -1
        elif self.interpolation_type == "gvp":
            return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t)
        elif self.interpolation_type == "gvp_w_sigma":
            return -self.sigma_dot_t(t) / torch.sqrt(
                1 - self.sigma_t(t) ** 2
            ) * torch.cos(0.5 * torch.pi * t) - 0.5 * torch.pi * torch.sin(
                0.5 * torch.pi * t
            )
        elif self.interpolation_type == "gvp_squared":
            return (
                -torch.pi
                * torch.cos(0.5 * torch.pi * t)
                * torch.sin(0.5 * torch.pi * t)
            )
    def f_t(self, t):
        return 1 / (2 * (1-t))

    def g_t(self, t):
        return 1 / (2 * t)

    def h_t(self, t):
        return (1 - 2*t) / (2 * t * (1 - t))

    def interpolate(self, x0, x1, t):
        return self.alpha_t(t) * x1 + self.beta_t(t) * x0

    def dtIt(self, x0, x1, t):
        return self.alpha_dot_t(t) * x1 + self.beta_dot_t(t) * x0

    def sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def sigma_dot_t(self, t):
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def sample_conditional_pt(
        self, x0: Tensor, x1: Tensor, t: Tensor, batch: Tensor, return_eps: bool = False
    ):
        # Have this here in case sample_conditional_pt
        # is used outside of compute_conditional_vector_field
        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # unsqueeze t and then reshape to number of atoms
        t = t[batch] if batch is not None else t
        t = unsqueeze_like(t, target=x0)

        # linear interpolation between x0 and x1
        # mu_t = self.interpolation_fn(x0, x1, t)
        eps = torch.randn_like(x1)

        # center each around center of mass
        eps = center_of_mass(eps, batch=batch)
        mu_t = self.interpolate(x0=x0, x1=x1, t=t)

        # no noise at t = 0 or t = 1
        x_t = mu_t + self.sigma_t(t) * eps

        if return_eps:
            return x_t, eps

        return x_t

    def compute_conditional_vector_field(self, x0, x1, t, batch=None):
        if batch is None:
            batch = torch.zeros((x1.size(0),)).to(self.device)

        # center both x0 and pos (x1: data distribution)
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)

        # sample a gaussian centered around the interpolation of x1, x0
        x_t, eps = self.sample_conditional_pt(x0, x1, t, batch=batch, return_eps=True)
        t = unsqueeze_like(t[batch], x1)

        # derivative of interpolate plus derivative of sigma function * noise
        if self.predict_mode == "v":
            if self.path_type == "cond_ot_path":
                coef = self.alpha_dot_t(t) / (1 - self.alpha_t(t))
                u_t = (
                    coef * (x1 - self.interpolate(x0=x0, x1=x1, t=t))
                    + self.sigma_dot_t(t) * eps
                )
            else:
                u_t = self.dtIt(x0, x1, t) + self.sigma_dot_t(t) * eps
        elif self.predict_mode == "x":
            u_t = x1 
        else:
            raise NotImplementedError
        return x_t, u_t

    def compute_target(self, x0, x1, x1_all, t, all_perms, batch=None, x_t=None):
        '''
        TODO: Implement this function
        '''
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)
        if x_t is None:
            x_t, _ = self.sample_conditional_pt(x0, x1, t, batch=batch, return_eps=True)
        ori_t = t.clone()
        t = unsqueeze_like(t[batch], x1)
        
        # Number of nodes per graph
        num_nodes_per_graph = torch.bincount(batch).int()

        # Step 1: Create an offset mask to adjust indices for each graph
        graph_offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
        offsets = graph_offsets[batch]  # Shape: (num_nodes,)
        
        adjusted_permutations = all_perms + offsets.unsqueeze(1)
        
        # Step 2: Create a mask to filter out the padded -1
        valid_mask = all_perms != -1 # num_nodes x num_perms
        invalid_mask = ~valid_mask # num_nodes x num_perms

        num_rots = self.num_rotations

        if num_rots == 0:
            num_rots = 1
        rots = random_rotation_matrices(num_rots, device=self.device, reflect=False) # n_r x 3 x 3
        rots[0] = torch.eye(3, device=self.device)
        n1 = x1.shape[0]
        n1_all = x1_all.shape[0]
        n_pos_per_smile = n1_all // n1
        if n_pos_per_smile == 1:
            # Compute offsets
            offsets = torch.tensor([0] + list(torch.cumsum(num_nodes_per_graph, dim=0) * n_pos_per_smile), device=self.device)
            # Split tensor into groups based on N_i
            split_tensors = [x1_all[offsets[i]:offsets[i+1]] for i in range(len(num_nodes_per_graph))]
            # Reshape each group to (n, Ni, d)
            split_tensors = [t.view(n_pos_per_smile, num_nodes_per_graph[i], 3) for i, t in enumerate(split_tensors)]
            # Stack groups per sample to create 2D matrices
            stacked_samples = [torch.cat([split_tensors[j][i] for j in range(len(num_nodes_per_graph))], dim=0) for i in range(n_pos_per_smile)]
            # Stack all samples to create a 3D tensor
            x1_all = torch.stack(stacked_samples, dim=0)
        else:
            x1_all = x1
        logits = []
        rotated_permuted_x1s = []
        for i in range(n_pos_per_smile): # for each conformer! Only the first conformer is similar to x1
            x1i = x1_all[i]
            x1i = center_of_mass(x1i, batch=batch)

            permuted_x1 = x1i[adjusted_permutations.T] # n_p x n_n x 3
            rotated_permuted_x1 = torch.einsum('ijp,kpq->kijp', permuted_x1, rots) # n_r x n_p x n_n x 3        

            #i, N, j 1, p 3 
            # rots cux: 1000, p 3, q 3 -> k 1000, N, 
            rotated_permuted_x1s.append(rotated_permuted_x1)

            # Compute the logit
            t_expanded = t.unsqueeze(0).unsqueeze(0)
            
            tx1 = t_expanded * rotated_permuted_x1
            
            x_t_expanded = x_t.unsqueeze(0).unsqueeze(0)
                        
            diff = tx1 - x_t_expanded # t * x_1 - x_t
            diff_norm = -torch.sum(diff ** 2, dim=-1) # n_r x n_p x n_n
            invalid_mask_expanded = invalid_mask.T.expand(num_rots,-1,-1)
            diff_norm[invalid_mask_expanded] = float('-inf')

            graph_norm = scatter(diff_norm, batch, dim=2, reduce='sum') # n_r x n_p x n_g
            graph_norm_flat = graph_norm.permute(2, 0, 1).reshape(graph_norm.shape[2], -1)

            t_graph = scatter(t, batch, dim=0, reduce='mean')
            sigma_graph = self.sigma_t(t_graph)
            logit = 0.5 * graph_norm_flat / ((1-t_graph)**2 + sigma_graph**2)
            logits.append(logit)
        # logits list of n_pos x bs x n_p x n_r
        logits = torch.cat(logits, -1)
        if self.z is not None and type(self.z) == float:
            temperature = torch.clamp(self.z/(1-ori_t),max=1.0)
        else:
            temperature = 1.0
        weight = torch.nn.functional.softmax(logits/temperature, dim=-1) # 128 x 40K

        rotated_permuted_x1 = torch.cat(rotated_permuted_x1s, dim=0)
        rotated_permuted_x1_flat=rotated_permuted_x1.view(-1, rotated_permuted_x1.shape[2], 3)
        node_weights = weight[batch]
                
        node_weights_expanded = node_weights.T.unsqueeze(-1) # n_n x n_p * n_r x 1
        weighted_coords = rotated_permuted_x1_flat * node_weights_expanded
        
        A = weighted_coords.sum(dim=0)

        if self.predict_mode == 'v':
            ft = self.f_t(t)
            gt = self.g_t(t)
            ht = self.h_t(t)
            
            target = ht * x_t + ft * A - gt * x0
        elif self.predict_mode == 'x':
            target = A
        else:
            raise NotImplementedError
        return x_t, target

    def switch_parity_of_pos(
        self, pos, chiral_index, chiral_nbr_index, chiral_tag, batch
    ):
        assert all(
            [
                key is not None
                for key in [chiral_index, chiral_nbr_index, chiral_tag, batch]
            ]
        )
        num_graphs = batch.max().item() + 1
        sv = signed_volume(
            pos[chiral_nbr_index.view(chiral_index.shape[1], 4)].unsqueeze(2)
        ).squeeze()
        ct = chiral_tag
        z_flip = sv * ct

        graph_diag = torch.ones(num_graphs, device=self.device)
        graph_diag[batch[chiral_index][:, (z_flip == -1.0)].squeeze()] = -1.0
        node_factor = graph_diag[batch].unsqueeze(1)

        return pos * node_factor

    def sample_base_dist(
        self,
        size: torch.Size,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        smiles: Optional[str] = None,
    ):
        if self.prior_type == "gaussian":
            x0 = torch.randn(size=size, device=self.device)
        elif self.prior_type == "harmonic":
            assert (edge_index is not None) and (batch is not None)
            x0 = self.harmonic_sampler.sample(
                size=size, edge_index=edge_index, batch=batch, smiles=smiles
            ).to(self.device)

        return x0

    def sample_time(
        self,
        num_samples: int,
        low: float = 1e-4,
        high: float = 0.9999,
        stage: str = "train",
    ):
        # batch_size = batch.max().item() + 1
        if isinstance(self.sample_time_dist, float) and 0.0 <= self.sample_time_dist <= 1.0:
            return torch.full((num_samples, 1), self.sample_time_dist, device=self.device)
        if self.sample_time_dist == "uniform" or stage == "val":
            # TODO: remove this later on, to remain consistent with val metrics
            # clamp to ensure numerical stability
            return torch.zeros(size=(num_samples, 1), device=self.device).uniform_(
                low, high
            )
        elif self.sample_time_dist == "logit_norm":
            return torch.sigmoid(torch.randn(size=(num_samples, 1))).to(self.device)
        else:
            raise NotImplementedError(
                f"Sample time distribution {self.sample_time_dist} not implemented"
            )

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        bond_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ):
        # center the positions at 0
        pos = center_of_mass(pos, batch=batch)

        # compute extended bond index
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=bond_index,
            batch=batch,
            bond_attr=edge_attr,
            device=self.device,
            one_hot=self.edge_one_hot,
            one_hot_types=self.edge_one_hot_types,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )

        # compute energy and score from network
        v_t = self.network(
            z=z,
            t=t[batch],
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_type,
            node_attr=node_attr,
            batch=batch,
        )

        return v_t

    def generic_step(self, batched_data, batch_idx: int, stage: str):
        # atomic numbers
        z, pos, bond_index, node_attr, edge_attr, all_perms, batch, all_pos = (
            batched_data["atomic_numbers"],
            batched_data["pos"],
            batched_data["edge_index"],
            batched_data.get("node_attr", None),  # optional
            batched_data.get("edge_attr", None),  # optional
            batched_data.get("all_perms", None),  # optional
            batched_data.get("batch", None),  # optional
            batched_data.get("all_pos", None),  # optional
        )
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # sample base distribution, either from harmonic or gaussian
        # x0 is sampling distribution and not data distribution
        x0 = self.sample_base_dist(
            pos.shape,
            edge_index=bond_index,
            batch=batch,
            smiles=batched_data.get("smiles", None),
        )

        # sample time steps equal to number of molecules in a batch
        t = self.sample_time(num_samples=batch_size, stage=stage)

        if self.prior_type == "harmonic":
            x0 = rmsd_align(pos=x0, ref_pos=pos, batch=batch)

        # check if x0 is nan
        if torch.isnan(x0).any():
            raise ValueError("x0 is NaN. Fix bug in harmonic alignment!")
        
        if self.train_mode == "ief" or self.train_mode == "min":
            x_t, u_t = self.compute_target(x0=x0, x1=pos, x1_all=all_pos, t=t, all_perms=all_perms, batch=batch)
        else:
            x_t, u_t = self.compute_conditional_vector_field(
                x0=x0, x1=pos, t=t, batch=batch
            )
        v_t = self(
            z=z,
            t=t,
            pos=x_t,
            bond_index=bond_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            batch=batch,
        )
        loss = batchwise_l2_loss(v_t, u_t, batch=batch, reduce="mean")
        if torch.isnan(loss):
            raise ValueError("Loss is NaN, fix bug")

        # log loss
        self.log_step_helper(f"{stage}/flow_matching_loss", loss, batch_size=batch_size)
        self.log_step_helper(f"{stage}/loss", loss, batch_size=batch_size)

        return loss

    def _compute_delta_t(self, t_schedule: Tensor, t: Tensor):
        if t + 1 >= t_schedule.size(0):
            return 0.0

        t_curr, t_next = t_schedule[t : t + 2]
        return t_next - t_curr

    def log_step_helper(self, key: str, value: torch.Tensor, batch_size: int):
        self.log(
            key,
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        
    @torch.no_grad()
    def sample(
        self,
        z: Tensor,
        bond_index: Tensor,
        batch: Tensor,
        node_attr: Tensor = None,
        edge_attr: Tensor = None,
        chiral_index: Tensor = None,
        chiral_nbr_index: Tensor = None,
        chiral_tag: Tensor = None,
        n_timesteps: int = 50,
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
    ):
        """
        By default performs ODE (sampler_type="ode") sampling
        If sampler_type is set to "stochastic", then it performs stochastic sampling
        """
        assert sampler_type == "ode", "Only support ode"
        t_schedule = torch.linspace(0, 1.0, steps=n_timesteps + 1, device=self.device)

        x = center_of_mass(
            self.sample_base_dist((z.size(0), 3), bond_index, batch), batch=batch
        )
        gamma = torch.tensor(s_churn / n_timesteps).to(self.device)

        n = t_schedule.size(0) - 1
        for i in range(n):
            t = t_schedule[i].repeat(x.size(0))
            t = unsqueeze_like(t, x)
            delta_t = self._compute_delta_t(t_schedule, t=i)

            # We do ODE when t is outside of [s_min, s_max]
            if (
                t_schedule[i] < t_min or t_schedule[i] >= t_max
            ) or sampler_type == "ode":
                if self.predict_mode == "v":
                    v_t = self(
                        z=z,
                        t=t,
                        pos=x,
                        bond_index=bond_index,
                        edge_attr=edge_attr,
                        node_attr=node_attr,
                        batch=batch,
                    )
                    x = x + delta_t * v_t
                elif self.predict_mode == "x":
                    x1_hat = self(
                        z=z,
                        t=t,
                        pos=x,
                        bond_index=bond_index,
                        edge_attr=edge_attr,
                        node_attr=node_attr,
                        batch=batch,
                    )  
                    x0_hat = (x - t * x1_hat) /(1-t) 
                    s = t_schedule[i + 1].repeat(x.size(0),1)
                    x = s * x1_hat + (1-s) * x0_hat
                else:
                    raise NotImplementedError

            # Stochastic sampling
            else:
                # delta_hat = gamma*delta_t
                delta_hat = gamma * (1 - t_schedule[i])
                t_prev_int = t_schedule[i] - delta_hat
                t_prev = t_prev_int.repeat(x.size(0))
                t_prev = unsqueeze_like(t_prev, x)
                """linear noise"""
                sig_t_sq = t_schedule[i] ** 2
                sig_t_prev_sq = t_prev_int**2
                mean = torch.zeros_like(x)
                noise = torch.normal(mean=mean, std=std)
                noise = center_of_mass(noise, batch=batch)
                x_prev = (
                    x
                    + torch.sqrt(torch.abs(sig_t_sq - sig_t_prev_sq))
                    * noise
                    * delta_hat
                )  # quadratic + linear decay

                v_t_prev = self(
                    z=z,
                    t=t_prev,
                    pos=x_prev,
                    bond_index=bond_index,
                    edge_attr=edge_attr,
                    node_attr=node_attr,
                    batch=batch,
                )
                # update step
                x = x_prev + v_t_prev * (delta_t + delta_hat)

        if self.parity_switch == "post_hoc":
            x = self.switch_parity_of_pos(
                x, chiral_index, chiral_nbr_index, chiral_tag, batch
            )

        return x

    @torch.no_grad()
    def predict(
        self,
        smiles: List[str],
        max_batch_size: int = 1,
        num_samples: int = 1,
        n_timesteps=50,
        seed: int = 42,
        device: str = "cpu",
        s_churn: float = 1.0,
        t_min: float = 1.0,
        t_max: float = 1.0,
        std: float = 1.0,
        sampler_type: str = "ode",
        as_mol: bool = False,
    ):
        if seed is not None:
            seed_everything(seed)

        def sample(
            data,
            max_batch_size,
            num_samples,
            n_timesteps,
            device,
            s_churn,
            t_min,
            t_max,
            std,
            sampler_type,
        ):
            sampled_pos = []

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

                with torch.no_grad():
                    pos = self.sample(
                        z=z,
                        bond_index=edge_index,
                        batch=batch,
                        node_attr=node_attr,
                        n_timesteps=n_timesteps,
                        chiral_index=chiral_index,
                        chiral_nbr_index=chiral_nbr_index,
                        chiral_tag=chiral_tag,
                        s_churn=s_churn,
                        t_min=t_min,
                        t_max=t_max,
                        std=std,
                        sampler_type=sampler_type,
                    )

                # reshape to (num_samples, num_atoms, 3) using batch
                pos = pos.view(batch_size, -1, 3).cpu().detach().numpy()

                # append to generated_positions
                sampled_pos.append(pos)

            # concatenate generated_positions
            sampled_pos = np.concatenate(
                sampled_pos, axis=0
            )  # (num_samples, num_atoms, 3)

            return sampled_pos

        feat = MoleculeFeaturizer()
        if not isinstance(smiles, list):
            smiles = [smiles]

        data = {}
        for smile in smiles:
            pos = sample(
                feat.get_data_from_smiles(
                    smile,
                ),
                max_batch_size,
                num_samples,
                n_timesteps,
                device,
                s_churn,
                t_min,
                t_max,
                std,
                sampler_type,
            )
            if as_mol:
                mol = get_mol_from_smiles(smile)
                data[smile] = set_multiple_rdmol_positions(mol, pos)
            else:
                data[smile] = pos
        return data


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
