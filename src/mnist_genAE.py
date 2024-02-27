'''
    Train generative GeoAE on MNIST dataset
'''
import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torchvision
from transformations import StandardScaler, MinMaxScaler, PowerTransformer, LogTransform, NonTransform


from model import MLP


''' GeoAE Model '''
class GeoAE(torch.nn.Module):
    def __init__(self, dim, emb_dim, 
                 layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), 
                 dist_reconstr_weights=[0.9, 0.1],
                 preprocessing=None,
                 eps=1e-8):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.dist_reconstr_weights = dist_reconstr_weights

        self.encoder = MLP(dim, emb_dim, 
                           layer_widths=layer_widths, 
                           activation_fn=activation_fn)
        self.decoder = MLP(emb_dim, dim, 
                           layer_widths=layer_widths[::-1], 
                           activation_fn=activation_fn)
        
        
        self.eps = eps

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        '''
            Returns:
                x_hat: [B, D]
                z: [B, emb_dim]
        '''
        z = self.encode(x)
        return [self.decode(z), z]
    
    def decoder_loss(self, x, x_hat):
        '''
            x: [B, D]
            x_hat: [B, D]
        '''
        return torch.nn.functional.mse_loss(x, x_hat)
    
    def encoder_loss(self, z, gt_dist):
        '''
            Inputs:
               z: [B, emb_dim]
               gt_dist: [B, B(B-1)/2] (upper triangular), assuming symmetric distance matrix
            Returns:
                loss: scalar
        '''
        pred_dist = torch.nn.functional.pdist(z, p=2)
        if self.preprocessing is not None:
            gt_dist = self.preprocessing.transform(gt_dist)
        mse_loss = torch.nn.functional.mse_loss(gt_dist, pred_dist, reduction='none')

        loss = torch.mean(torch.mean(mse_loss, dim=1))

        return loss
    

class GenerativeGeoAE(torch.nn.Module):
    ''' Generative GeoAE '''
    def __init__(self, dim, emb_dim, time_embed_dim,
                 layer_widths=[64, 64, 64], activation_fn=torch.nn.ReLU(), 
                 autoencoder: torch.nn.Module=None,
                 dropout=0.0, batch_norm=False,
                 eps=1e-8):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.time_embed_dim = time_embed_dim
        self.autoencoder = autoencoder

        # check autoencoder has encode and decode methods
        assert hasattr(autoencoder, 'encode') and hasattr(autoencoder, 'decode')
        
        self.time_embedding = nn.Linear(1, time_embed_dim)
        self.network = MLP(dim = dim + time_embed_dim,
                           out_dim = emb_dim,
                           layer_widths=layer_widths,
                           activation_fn=activation_fn,
                           dropout=dropout,
                           batch_norm=batch_norm
                        )
        
        self.eps = eps

    def forward(self, x, t):
        '''
            predict the noise at time t, given x
        '''
        # Ensure t is a 2D tensor with shape [batch_size, 1]
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        
        # Embedding the time step
        t_emb = self.time_embedding(t)

        # Concatenating the time embedding to the input
        xt = torch.cat([x, t_emb], dim=1) # [B, D + time_embed_dim]
        
        return self.network(xt)
    
    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)
    
    def generate_latent(self, num_samples, num_steps):
        '''
            Generate samples from the latent space
            Inputs:
                num_samples: int
                num_steps: int
            Returns:
                x_t: [num_samples, D]
        '''
        self.network.eval()  # Ensure the model is in evaluation mode.

        # Initial random noise
        x_t = torch.randn(num_samples, self.emb_dim)

        # Linear noise schedule used during training
        betas = torch.linspace(0.0001, 0.02, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Generate t values for reverse process, normalized
        t_values = torch.arange(0, num_steps).flip(0) #[num_steps, ..., 1]

        with torch.no_grad():
            for t in t_values:
                # Calculate the current alpha and noise for this step
                sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
                sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t])
                t_norm = t.float() / (num_steps - 1)
                t_batch = t_norm.repeat(num_samples, 1)
                # Prepare t for model input (ensure it's the correct shape and type)

                # Model prediction to refine the current state
                epsilon_t = self.network(x_t, t_batch)
                mu_t = (x_t - (1 - alphas[t]) / sqrt_one_minus_alphas_cumprod * epsilon_t) / torch.sqrt(alphas[t])
                noise1 = torch.randn(num_samples, self.emb_dim)
                x_t = mu_t + torch.sqrt(betas[t]) * noise1

        return x_t
    
from data import PointCloudDataset
import phate
import scipy 

def train(cfg: OmegaConf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor_dict = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'power': PowerTransformer(),
                'log': LogTransform(),
    }
    preprocessor = preprocessor_dict[cfg.model.preprocessing]
    # check if the npy files exist
    if os.path.exists('./test_genmnist/flattened_mnist.npy'):
        flatten_mnist = np.load('./test_genmnist/flattened_mnist.npy')
        phate_embed = np.load('./test_genmnist/phate_embedding.npy')
        gt_dist = np.load('./test_genmnist/gt_matrix.npy')
    else:
        ''' Load Data '''
        mnist_data = torchvision.datasets.MNIST(root=f'{cfg.data.root}/{cfg.data.name}',
                                                train=False, download=True, 
                                                transform=torchvision.transforms.ToTensor())
        # Phate Distance
        print('preprocessor:', preprocessor)
        flatten_mnist = mnist_data.data.numpy().reshape(-1, 28*28)
        phate_op = phate.PHATE(random_state=1, 
                            verbose=True,
                            n_components=cfg.model.emb_dim,
                            knn=5).fit(flatten_mnist)
        phate_embed = phate_op.transform(flatten_mnist)
        diff_pot = phate_op.diff_potential
        os.makedirs('./test_genmnist', exist_ok=True)
        np.save('./test_genmnist/flattened_mnist.npy', flatten_mnist)
        np.save('./test_genmnist/phate_embedding.npy', phate_embed)

        if cfg.model.dist_type == 'phate':
            gt_dist = scipy.spatial.distance.cdist(diff_pot, diff_pot)
            gt_dist = preprocessor.fit_transform(gt_dist)
        elif cfg.model.dist_type == 'diffusion_potential':
            diff_pot = preprocessor.fit_transform(diff_pot)
        gt_dist = scipy.spatial.distance.cdist(diff_pot, diff_pot)
        # Save flattened mnist, phate embedding and distance matrix
        np.save('./test_genmnist/gt_matrix.npy', gt_dist)

    print('flatttened mnist:', flatten_mnist.shape)
    print('gt dist:', gt_dist.shape)
    pointcloud_dataset = PointCloudDataset(flatten_mnist, distances=gt_dist, 
                                           batch_size=cfg.training.batch_size, shuffle=True)
    mnist_loader = torch.utils.data.DataLoader(pointcloud_dataset, 
                                               batch_size=None, 
                                               shuffle=True)
    
    print(mnist_loader)

    ''' Model '''
    autoencoder = GeoAE(dim=28*28, emb_dim=cfg.model.emb_dim, 
                  layer_widths=cfg.model.layer_widths, 
                  activation_fn=torch.nn.ReLU(), 
                  preprocessing=preprocessor_dict[cfg.model.preprocessing],
                  dist_reconstr_weights=cfg.model.dist_reconstr_weights)
    model = GenerativeGeoAE(dim=28*28, emb_dim=cfg.model.emb_dim, time_embed_dim=cfg.model.time_emb_dim,
                            layer_widths=cfg.model.layer_widths, activation_fn=torch.nn.ReLU(), 
                            autoencoder=autoencoder,
                            dropout=cfg.model.dropout, 
                            batch_norm=cfg.model.batch_norm)

    ''' Optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    ''' Train '''
    latent_data = []
    for epoch in range(cfg.training.max_epochs):
        for idx, batch in enumerate(mnist_loader):
            x = batch['x']
            batch_D = batch['d']
            print('Batch:', idx, x, batch_D)
            x = x.view(-1, 28*28).to(device) # flatten [B, 28, 28] -> [B, 28*28]
            batch_D = batch_D.to(device)

            optimizer.zero_grad()

            # Forward
            print(x.shape, batch_D.shape)
            z = model.autoencoder.encode(x)
            x_hat = model.autoencoder.decode(z)

            if epoch == cfg.training.max_epochs - 1:
                latent_data.append(z.detach().cpu().numpy())
            
            # Loss
            decoder_loss = model.autoencoder.decoder_loss(x, x_hat)
            encoder_loss = model.autoencoder.encoder_loss(z, batch_D)
            loss = decoder_loss * model.dist_reconstr_weights[1] \
                + encoder_loss * model.dist_reconstr_weights[0]

            # Update
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

    ''' Save Model '''
    os.makedirs('./test_genmnist/', exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'model.pth'))

    # Get all latent embeddings
    latent_data = np.concatenate(latent_data, axis=0)
    latent_dataloader = torch.utils.data.DataLoader(latent_data, 
                                                    batch_size=cfg.training.batch_size, 
                                                    shuffle=False)
    ''' Train Diffusion Model '''
    # Train the diffusion model
    print('Training diffusion model...')
    model.train()

    # Example linear noise schedule (beta values)
    num_steps = 100
    betas = torch.linspace(0.0001, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(cfg.training.max_epochs):
        for batch in latent_dataloader:
            data = batch[0].to(device)
            print(data.shape)
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
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                loss.backward()
                optimizer.step()

        print(f"DM Epoch: {epoch}, Loss: {loss.item()}")
    
    # Generate samples from the trained model
    num_samples = 10
    num_steps = 100
    generated_samples = model.generate_latent(num_samples, num_steps)
    print(generated_samples.shape)

    # Visualize the generated samples
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(generated_samples.cpu().numpy())
    # ax.set_title('Generated Latent Space')

    # decode the generated samples
    generated_data = model.decode(generated_samples)
    print(generated_data.shape)

    # Visualize the generated data
    import matplotlib.pyplot as plt
    for i in range(num_samples):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(generated_data[i].cpu().numpy().reshape(28, 28))
        ax.set_title(f'Generated Data {i}')
        # save the generated data
        plt.savefig(f'./test_genmnist/generated_data_{i}.png')


if __name__ == '__main__':
    cfg = OmegaConf.load('../conf/mnist_config.yaml')
    train(cfg)