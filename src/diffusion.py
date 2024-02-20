import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MLP

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiffusionModel(nn.Module):
    def __init__(self, data_size=100, time_embedding_size=20, layer_widths=[64,64,64],activation_fn=torch.nn.ReLU(),dropout=0.0, batch_norm=False):
        super(DiffusionModel, self).__init__()
        self.time_embedding = nn.Linear(1, time_embedding_size)
        self.network = MLP(
                           dim = data_size + time_embedding_size,
                           out_dim = data_size,
                           layer_widths=layer_widths,
                           activation_fn=activation_fn,
                           dropout=dropout,
                           batch_norm=batch_norm
                        )

    def forward(self, x, t):
        # Ensure t is a 2D tensor with shape [batch_size, 1]
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        
        # Embedding the time step
        t_emb = self.time_embedding(t)

        # Concatenating the time embedding to the input
        xt = torch.cat([x, t_emb], dim=1)
        
        return self.network(xt)


def train_diffusion_model(model, data_loader, optimizer, num_epochs, num_steps, device):
    model.train()
    
    # Example linear noise schedule (beta values)
    betas = torch.linspace(0.0001, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(num_epochs):
        for batch in data_loader:
            data = batch[0].to(device)
            
            for step in range(1, num_steps + 1):
                # Sample time step
                t = torch.randint(0, num_steps, (data.size(0),), device=device)
                t_float = t.float() / (num_steps - 1)
                
                # Calculate noise for this step
                noise = torch.randn_like(data)
                
                # Calculate the noisy data based on the noise schedule
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t]).unsqueeze(-1)
                noisy_data = sqrt_alphas_cumprod * data + sqrt_one_minus_alphas_cumprod * noise
                
                optimizer.zero_grad()
                
                # Model prediction (reverse process)
                predicted_noise = model(noisy_data, t_float)
                
                # Calculate loss as MSE between the predicted and actual noise
                loss = F.mse_loss(predicted_noise, noise)
                
                loss.backward()
                optimizer.step()
                
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

def generate_samples(model, num_samples, data_size, num_steps, device='cpu'):
    model.eval()  # Ensure the model is in evaluation mode.

    # Initial random noise
    x_t = torch.randn(num_samples, data_size).to(device)

    # Linear noise schedule used during training
    betas = torch.linspace(0.0001, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Generate t values for reverse process, normalized
    t_values = torch.arange(0, num_steps).flip(0).to(device)

    with torch.no_grad():
        for t in t_values:
            # Calculate the current alpha and noise for this step
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])
            t_norm = t.float() / (num_steps - 1)
            t_batch = t_norm.repeat(num_samples, 1)
            # Prepare t for model input (ensure it's the correct shape and type)

            # Model prediction to refine the current state
            epsilon_t = model(x_t, t_batch)
            mu_t = (x_t - (1 - alphas[t]) / sqrt_one_minus_alphas_cumprod * epsilon_t) / torch.sqrt(alphas[t])
            noise1 = torch.randn(num_samples, data_size).to(device)
            x_t = mu_t + torch.sqrt(betas[t]) * noise1

    return x_t
