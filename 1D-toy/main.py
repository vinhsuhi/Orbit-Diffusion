import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random 
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
import os

# Set Random Seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Initialize Data
def initialize_data():
    data = torch.ones(200, 1).cuda()
    return data

# Initialize Sigmas and Alphas
def initialize_diffusion_params(num_timesteps=100):
    sigmas = torch.linspace(0.01, 0.999, num_timesteps).cuda()
    alphas = torch.sqrt(1 - sigmas ** 2)
    return sigmas, alphas


def forward_process(x, t, alphas, sigmas):
    noise = torch.randn_like(x).cuda()
    return alphas[t] * x + sigmas[t] * noise, noise


# Sinusoidal Time Embedding
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_timesteps=1000):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        emb = emb * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1)).exp()
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# Denoiser with Time Embedding
class EquiDenoiserWithTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.Tanh(),
            nn.Linear(10, 5), nn.Tanh(),
            nn.Linear(5, 1)
        )

    def forward(self, x, t):
        x_input = x
        t_embed = t
        x_t_embed = torch.cat([x_input, t_embed], dim=-1)
        neg_x_t_embed = torch.cat([-x_input, t_embed], dim=-1)
        out_x = self.net(x_t_embed)
        out_neg_x = self.net(neg_x_t_embed)
        return out_x - out_neg_x


# Define a simple denoising model with sinusoidal time embedding
class NonEquiDenoiserWithTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(2, 10),  # Input includes time embedding + noisy data
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 1)  # Predict noise
        )

    def forward(self, x, t): # equivariant 
        # x_input = x.unsqueeze(-1)
        # t_embed = self.time_embedding(t)  # Compute time embeddings
        x_input = x
        t_embed = t
        x_t_embed = torch.cat([x_input, t_embed], dim=-1)  # Concatenate x and t_embed
        out_x = self.net(x_t_embed)  # Predict noise
        return out_x
    
    
# Sampling Function
@torch.no_grad()
def sample(model, x, alphas, sigmas, num_timesteps):
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t / num_timesteps], dtype=torch.float32).cuda()
        t_tensor = t_tensor.expand(x.shape)

        alpha_t = alphas[t]
        sigma_t = sigmas[t]

        denoised = model(x, t_tensor)
        predicted_noise = (x - alpha_t * denoised) / sigma_t

        if t > 0:
            alpha_s = alphas[t - 1]
            sigma_s = sigmas[t - 1]
            x = alpha_s * denoised + sigma_s * predicted_noise

    return denoised

# Training Function
def train_diffusion_model(equi, baseline, augment, temperature=-1, argmax=False, num_epochs=20000, num_timesteps=100, sample_size=1000, seed=42, save_path='./'):
    set_seed(seed)
    data = initialize_data()
    val_noises = torch.abs(torch.randn(sample_size, 1)).cuda()
    sigmas, alphas = initialize_diffusion_params(num_timesteps)
    if equi:
        model = EquiDenoiserWithTimeEmbedding().cuda()
    else:
        model = NonEquiDenoiserWithTimeEmbedding().cuda()
    print(sum(p.numel() for p in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_schedule = StepLR(optimizer, step_size=8000, gamma=0.8)

    best_error = float("inf")
    losses, errors = [], []
    n = 0
    for epoch in tqdm(range(num_epochs)):
        t = torch.randint(0, num_timesteps, data.shape).cuda()
        if augment:
            data = torch.randint(0, 2, data.shape).cuda() * 2 - 1
            data = data.float()
        
        noisy_x, noise = forward_process(data, t, alphas, sigmas)

        t_tensor = t / num_timesteps

        denoised = model(noisy_x, t_tensor)
        if baseline:
            target = data
        else:
            diff_1 = data * alphas[t] - noisy_x
            diff_2 = -data * alphas[t] - noisy_x 
            
            dist_1 = torch.abs(diff_1) ** 2
            dist_2 = torch.abs(diff_2) ** 2
            
            weight_1 = -dist_1 / (2 * sigmas[t] ** 2)
            weight_2 = -dist_2 / (2 * sigmas[t] ** 2)
            
            if temperature > 0:
                weight_1 = weight_1 / temperature
                weight_2 = weight_2 / temperature
            weight = torch.softmax(torch.cat([weight_1, weight_2], dim=-1), dim=-1)
            

            weight_1 = weight[:, 0].reshape(-1, 1)
            weight_2 = weight[:, 1].reshape(-1, 1)
            if argmax:
                weight_1 = (weight_1 > weight_2).float()
                weight_2 = 1 - weight_1
            target = weight_1 * data + weight_2 * -data
        
        loss = ((denoised - target) ** 2) * alphas[t] ** 2 / sigmas[t] ** 2
        loss = loss.mean()
        n += torch.sum(torch.sign(noise) != torch.sign(target)).item()
        # loss = ((denoised - data) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()

        if epoch % 500 == 0 and epoch > 1:

            losses.append(loss.item())
            _, error = evaluate_model(model, val_noises, alphas, sigmas, num_timesteps)
            errors.append(error.item())
            
            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Error: {error}")
                print(f"learning rate: {lr_schedule.get_last_lr()}")

            if error <= best_error:
                best_error = error
                torch.save(model, os.path.join(save_path, f"model_{seed}.pt"))
    print(n)
    return losses, errors
      

# Evaluation Function
def evaluate_model(model, val_noises, alphas, sigmas, num_timesteps):
    generated_samples = sample(model, val_noises, alphas, sigmas, num_timesteps)
    error = torch.abs(generated_samples[generated_samples > 0.] - 1.0).mean()

    return generated_samples, error


# Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate a diffusion model.")
    parser.add_argument('--num_epochs', type=int, default=100000, help="Number of training epochs")
    parser.add_argument('--num_timesteps', type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument('--num_seeds', type=int, default=3, help="Number of seeds for reproducibility")
    parser.add_argument('--save_path', type=str, default='trained_models/', help="Directory to save trained models")
    parser.add_argument('--equi', action="store_true")
    parser.add_argument('--baseline', action="store_true")
    parser.add_argument('--augment', action="store_true")
    parser.add_argument('--temperature', type=float, default=-1, help="temperature")
    parser.add_argument('--argmax',  action="store_true")

    args = parser.parse_args()
    exp_name = "baseline" if args.baseline else "ours"
    exp_name += "_equi" if args.equi else "_nonequi"
    exp_name += "_augment" if args.augment else ""
    if args.temperature > 0:
        exp_name += f"_temp{args.temperature}"
    exp_name += f"_argmax" if args.argmax else ""
    args.save_path = os.path.join(args.save_path, exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    sigmas, alphas = initialize_diffusion_params(args.num_timesteps)
    losses_seed,  errors_seed = [], []
    
    
    for seed in range(args.num_seeds):
        losses, errors = train_diffusion_model(
                        equi=args.equi,
                        baseline=args.baseline,
                        augment=args.augment,
                        temperature=args.temperature,
                        argmax=args.argmax,
                        num_epochs=args.num_epochs,
                        num_timesteps=args.num_timesteps, 
                        seed=seed,
                        save_path=args.save_path,
                        )
        errors_seed.append(errors)
        losses_seed.append(losses)
    
    errors_array = np.array(errors_seed)  # Shape: (3, num_epochs)
    losses_array = np.array(losses_seed)  # Shape: (3, num_epochs)

    np.save(os.path.join(args.save_path, f"validation_error.npy"), errors_array)
    np.save(os.path.join(args.save_path, f"training_loss.npy"), losses_array)
