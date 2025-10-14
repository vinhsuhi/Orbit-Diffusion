
import numpy as np
import torch


def random_rotation_matrices(num_matrices, device='cpu', reflect=False):
    # Generate random numbers directly on the specified device
    u1 = torch.rand(num_matrices, device=device)
    u2 = torch.rand(num_matrices, device=device)
    u3 = torch.rand(num_matrices, device=device)

    # Compute quaternions
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # Compute the rotation matrices using broadcasting
    Rs = torch.empty((num_matrices, 3, 3), device=device)
    Rs[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    Rs[:, 0, 1] = 2 * (q2 * q3 - q1 * q4)
    Rs[:, 0, 2] = 2 * (q2 * q4 + q1 * q3)
    Rs[:, 1, 0] = 2 * (q2 * q3 + q1 * q4)
    Rs[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    Rs[:, 1, 2] = 2 * (q3 * q4 - q1 * q2)
    Rs[:, 2, 0] = 2 * (q2 * q4 - q1 * q3)
    Rs[:, 2, 1] = 2 * (q3 * q4 + q1 * q2)
    Rs[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)
    
    if reflect:
        mask = torch.rand(num_matrices, device=device) < 0.5
        Rs[mask, :, 0] *= -1

    return Rs



def kabsch_algorithm(P, Q):
    """
    Compute the optimal rotation matrix using the Kabsch algorithm.
    
    Parameters:
    P: ndarray of shape (N, 3), the first set of points
    Q: ndarray of shape (N, 3), the second set of points
    
    Returns:
    R: the optimal rotation matrix
    """
    # Center the points
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    # Covariance matrix
    C = np.dot(P_centered.T, Q_centered)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(C)

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R

def compute_alignment_distance(P: torch.Tensor, Q: torch.Tensor):
    """
    Compute the alignment distance after applying the Kabsch algorithm.
    
    Parameters:
    P: ndarray of shape (N, 3), the first set of points
    Q: ndarray of shape (N, 3), the second set of points
    
    Returns:
    distance: the mean squared distance after alignment
    P should be a clean sample, Q should be a noisy sample
    """
    device = P.device
    P = P.cpu().numpy()
    Q = Q.cpu().numpy()
    R = kabsch_algorithm(P, Q)
    
    # Apply the rotation matrix
    P_rotated = np.dot(P, R)
    # Compute the mean squared distance
    distance = np.mean(np.sum((P_rotated - Q) ** 2, axis=1))
    
    return torch.from_numpy(R).to(device), distance, P_rotated


def compute_logit_parallel(x, alpha_t, sigma_t, mean):
    # x: n_s x bs x n x 3 (yes)
    # mean: bs x n x 3 (no!) 
    # alpha_t: bs x 1 x 1 (yes)
    # sigma_t: bs x 1 x 1 (yes)
    x_flat = x.reshape(x.shape[0], x.shape[1], -1) # n_s x bs x n * 3
    mean_flat = mean.reshape(x.shape[1], -1).unsqueeze(0) # 1 x bs x n * 3
    alpha_t = alpha_t.reshape(x.shape[1], 1).unsqueeze(0) # 1 x bs x 1
    sigma_t = sigma_t.reshape(x.shape[1], 1).unsqueeze(0) # 1 x bs x 1
    square_distance = (alpha_t * x_flat - mean_flat) ** 2
    logit = -0.5 * square_distance.sum(dim=-1, keepdim=True) / sigma_t ** 2
    return logit.reshape(logit.shape[0], -1)
