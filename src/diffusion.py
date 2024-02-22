import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import MLP

class DiffusionModel(pl.LightningModule):
    def __init__(self, data_size=100, time_embedding_size=20, layer_widths=[64, 64, 64], activation_fn=nn.ReLU(), dropout=0.0, batch_norm=False, num_steps=1000, learning_rate=1e-3, weight_decay=1e-4):
        super(DiffusionModel, self).__init__()
        self.save_hyperparameters()
        
        self.time_embedding = nn.Linear(1, self.hparams.time_embedding_size)
        self.network = MLP(
            dim=self.hparams.data_size + self.hparams.time_embedding_size,
            out_dim=self.hparams.data_size,
            layer_widths=self.hparams.layer_widths,
            activation_fn=self.hparams.activation_fn,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm
        )

        # Prepare the betas, alphas, and alphas_cumprod for use in training and sampling
        betas = torch.linspace(0.0001, 0.02, self.hparams.num_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def forward(self, x, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        t_emb = self.time_embedding(t)
        xt = torch.cat([x, t_emb], dim=1)
        return self.network(xt)

    def training_step(self, batch, batch_idx):
        data = batch[0]  # Assuming your DataLoader provides inputs and targets
        device = data.device
        num_steps = self.hparams.num_steps
        t = torch.randint(0, num_steps, (data.size(0),), device=device)
        t_float = t.float() / (num_steps - 1)
        noise = torch.randn_like(data)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).unsqueeze(-1)
        noisy_data = sqrt_alphas_cumprod * data + sqrt_one_minus_alphas_cumprod * noise
        predicted_noise = self(noisy_data, t_float)
        loss = F.mse_loss(predicted_noise, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def generate_samples(self, num_samples):
        self.eval()  # Ensure the model is in evaluation mode

        # Initial random noise
        x_t = torch.randn(num_samples, self.hparams.data_size, device=self.device)

        # Utilize the buffers for betas, alphas, and alphas_cumprod (already on the correct device)
        betas = self.betas
        alphas = self.alphas
        alphas_cumprod = self.alphas_cumprod

        # Generate t values for reverse process, normalized
        t_values = torch.arange(0, self.hparams.num_steps, device=self.device).flip(0)

        with torch.no_grad():
            for t in t_values:
                # Calculate the current alpha and noise for this step
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])
                t_norm = t.float() / (self.hparams.num_steps - 1)
                t_batch = t_norm.repeat(num_samples, 1)

                # Model prediction to refine the current state
                epsilon_t = self(x_t, t_batch)
                mu_t = (x_t - (1 - alphas[t]) / sqrt_one_minus_alphas_cumprod * epsilon_t) / torch.sqrt(alphas[t])
                noise1 = torch.randn(num_samples, self.hparams.data_size, device=self.device)
                x_t = mu_t + torch.sqrt(betas[t]) * noise1

        return x_t

# Example usage:
# model = DiffusionModel()
# trainer = pl.Trainer(max_epochs=10, gpus=1)  # Adjust as per your setup
# trainer.fit(model, train_dataloader)
