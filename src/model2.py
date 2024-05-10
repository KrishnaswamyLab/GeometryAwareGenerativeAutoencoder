"""
Train encoder and decoder separately.
By the way refactor and clean up.
"""

import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from torch.nn.utils import spectral_norm

def calculate_bounding_radius(X, centroid):
    # centroid = torch.mean(X, dim=0)
    # Calculate distances from the centroid to all points
    distances = torch.norm(X - centroid, dim=1, p=2)  # Euclidean norm
    # Maximum distance from the centroid to any point
    bounding_radius = torch.max(distances)
    return bounding_radius

def generate_distant_points(centroid, bounding_radius, num_points, dim, distance_factor=1.5):
    # Generate random directions
    directions = torch.randn(num_points, dim, device=centroid.device, dtype=centroid.dtype)
    directions = directions / directions.norm(dim=1, keepdim=True)
    
    # Scale directions to have a radius that is beyond the bounding sphere
    scaled_radius = bounding_radius * distance_factor
    distant_points = centroid + directions * scaled_radius
    return distant_points

def generate_negative_samples(X, num_neg_samples, distance_factor=1.5):
    centroid = torch.mean(X, dim=0)
    bounding_radius = calculate_bounding_radius(X, centroid)
    num_points, dim = X.shape
    distant_points = generate_distant_points(centroid, bounding_radius, num_neg_samples, dim, distance_factor)
    return distant_points

activation_dict = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid()
}

class MLP(torch.nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        layer_widths = cfg.get("layer_widths", [64, 64, 64])
        assert len(layer_widths) >= 2, "layer_widths list must contain at least 2 elements"
        activation = cfg.get("activation", "relu")
        assert activation in activation_dict.keys(), f"activation must be one of {list(activation_dict.keys())}"
        batch_norm = cfg.get("batch_norm", False)
        dropout = cfg.get("dropout", 0.0)
        use_spectral_norm = cfg.get("spectral_norm", False)  # Configuration for using spectral normalization

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:  # First layer, input dimension to first layer width
                linear_layer = torch.nn.Linear(in_dim, width)
            else:  # Subsequent layers, previous layer width to current layer width
                linear_layer = torch.nn.Linear(layer_widths[i-1], width)

            # Conditionally apply spectral normalization
            if use_spectral_norm:
                linear_layer = spectral_norm(linear_layer)

            layers.append(linear_layer)

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(width))
            
            activation_func = activation_dict[activation]
            layers.append(activation_func)

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        
        # Adding the final layer
        final_linear_layer = torch.nn.Linear(layer_widths[-1], out_dim)
        if use_spectral_norm:
            final_linear_layer = spectral_norm(final_linear_layer)
        layers.append(final_linear_layer)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Encoder(pl.LightningModule):
    def __init__(self, cfg, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        out_dim = cfg.dimensions.get('latent')
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)

    def forward(self, x, normalize=True): # takes in unnormalized data.
        if normalize:
            x = self.preprocessor.normalize(x)
        return self.mlp(x)
    
    def negative_sampling_loss(self, x, negative_x, z, negative_z, margin=1.0, loss_type='triplet'):
        '''Negative sampling loss for contrastive learning'''
        loss = 0.0
        if loss_type == 'triplet':
            # distances between z and negative_z
            d_z_zneg = torch.nn.functional.pairwise_distance(z, negative_z, p=1)
            # distances between z and z
            d_z_z = torch.nn.functional.pdist(z, p=1)

            loss = torch.nn.functional.relu(d_z_z - d_z_zneg + margin).mean()
            # TODO: distances between z and negative_z should be larger than z and x?
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
    
        return loss
    
    def loss_function(self, dist_gt_norm, z, mask=None): # assume normalized.
        dist_emb = torch.nn.functional.pdist(z)
        if self.hparams.cfg.loss.dist_mse_decay > 0.:
            if mask is None:
                return (torch.square(dist_emb - dist_gt_norm) * torch.exp(-self.hparams.cfg.loss.dist_mse_decay * dist_gt_norm)).mean()
            else:
                return (torch.square(dist_emb - dist_gt_norm) * torch.exp(-self.hparams.cfg.loss.dist_mse_decay * dist_gt_norm) * mask).sum() / mask.sum()
        else:
            if mask is None:
                return torch.nn.functional.mse_loss(dist_emb, dist_gt_norm)
            else:
                return (torch.square(dist_emb - dist_gt_norm) * mask).sum() / mask.sum()

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        mask = batch.get('md', None)
        z = self.forward(x)
        d_norm = self.preprocessor.normalize_dist(d)
        loss = self.loss_function(d_norm, z, mask)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.cfg.training.lr, weight_decay=self.hparams.cfg.training.weight_decay)
        return optimizer


class Decoder(pl.LightningModule):
    def __init__(self, cfg, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        in_dim = cfg.dimensions.get('latent')
        out_dim = cfg.dimensions.get('data')
        self.save_hyperparameters()
        self.mlp = MLP(cfg.decoder, in_dim, out_dim)
        self.encoder = None
        self.use_encoder = False
    
    def set_encoder(self, encoder):
        self.encoder = encoder
        self.use_encoder = True

    def forward(self, z, unnormalize=True): # outputs unnormalized data
        x = self.mlp(z)
        if unnormalize:
            x = self.preprocessor.unnormalize(x)
        return x

    def loss_function(self, x_norm, xhat_norm, mask=None): # assume normalized.
        if mask is None:
            return torch.nn.functional.mse_loss(x_norm, xhat_norm)
        else:
            print('mx used')
            return (torch.square(x_norm - xhat_norm) * mask).sum() / mask.sum()

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None)
        if self.use_encoder:
            assert self.encoder is not None
            with torch.no_grad():
                z = self.encoder(x) # TODO check if this gives expected behavior.
        else:
            assert 'z' in batch.keys()
            z = batch['z'] # after encoder is trained, do we need to make a new dataset?
        x = self.preprocessor.normalize(x)
        xhat = self.forward(z, unnormalize=False)
        loss = self.loss_function(x, xhat, mask)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.cfg.training.lr, weight_decay=self.hparams.cfg.training.weight_decay)
        return optimizer

class Autoencoder(pl.LightningModule):
    def __init__(self, cfg, preprocessor):
        super().__init__()
        self.encoder = Encoder(cfg, preprocessor)
        self.decoder = Decoder(cfg, preprocessor)
        # self.save_hyperparameters(cfg)
        # self.save_hyperparameters(preprocessor_params)
        self.save_hyperparameters()

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def end2end_step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        mask_d = batch.get('md', None)
        mask_x = batch.get('mx', None)
        x_norm = self.encoder.preprocessor.normalize(x)
        zhat = self.encoder(x_norm)
        d_norm = self.encoder.preprocessor.normalize_dist(d)
        xhat_norm = self.decoder(zhat, unnormalize=False)
        loss = self.loss_function(xhat_norm, x_norm, zhat, d_norm, stage, mask_d, mask_x)
        return loss

    def loss_function(self, xhat_norm, x_norm, zhat, d_norm, stage, mask_d=None, mask_x=None):
        """output are the outputs of forward method"""
        # x, x_hat: [B, D]; z: [B, emb_dim]; gt_dist: [B, (B-1)/2]
        loss = 0.0
        assert self.hparams.cfg.loss.weights.dist + self.hparams.cfg.loss.weights.reconstr > 0.0, "At least one loss must be enabled"
        if self.hparams.cfg.loss.weights.dist > 0.0:
            dl = self.encoder.loss_function(d_norm, zhat, mask_d)
            self.log(f'{stage}/dist_loss', dl, prog_bar=True, on_epoch=True)
            loss += self.hparams.cfg.loss.weights.dist * dl

        if self.hparams.cfg.loss.weights.reconstr > 0.0:
            rl = self.decoder.loss_function(x_norm, xhat_norm, mask_x)
            self.log(f'{stage}/reconstr_loss', rl, prog_bar=True, on_epoch=True)
            loss += self.hparams.cfg.loss.weights.reconstr * rl

        if self.hparams.cfg.loss.weights.cycle + self.hparams.cfg.loss.weights.cycle_dist > 0.0:
            z2 = self.encoder(xhat_norm, normalize=False)
            if self.hparams.cfg.loss.weights.cycle > 0.0:
                if mask_x is None:
                    l2 = torch.nn.functional.mse_loss(zhat, z2)
                else:
                    l2 = (torch.square(zhat - z2) * mask_x).sum() / mask_x.sum()
                self.log(f'{stage}/cycle_loss', l2, prog_bar=True, on_epoch=True)
                loss += self.hparams.cfg.loss.weights.cycle * l2
            if self.hparams.cfg.loss.weights.cycle_dist > 0.0:
                l3 = self.encoder.loss_function(d_norm, z2, mask_d)
                self.log(f'{stage}/cycle_dist_loss', l3, prog_bar=True, on_epoch=True)
                loss += self.hparams.cfg.loss.weights.cycle_dist * l3
        return loss
    
        
    def negative_step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        mask_x = batch.get('mx', None).flatten().bool()
        mask_d = mask_x.reshape(-1,1) & mask_x.reshape(1,-1) 
        mask_d = mask_d[np.triu_indices(mask_d.size(0), k=1)]
        assert self.hparams.cfg.loss.weights.negative > 0.
        assert mask_x is not None, "Noisy negative sampling requires mask_x"
        x_norm = self.encoder.preprocessor.normalize(x)
        x_norm_p = x_norm[mask_x]
        zhat = self.encoder(x_norm)
        zp = zhat[mask_x]
        zn = zhat[~mask_x]
        zmeans = zp.mean(axis=0)
        zstds = zp.std(axis=0)
        # noise = torch.randn_like(zn) * zstds * 3. + zmeans
        noise = generate_negative_samples(zp, zn.size(0))
        dp = d[mask_d]
        d_norm_p = self.encoder.preprocessor.normalize_dist(dp)
        xhat_norm_p = self.decoder(zp, unnormalize=False)
        loss_p = self.loss_function(xhat_norm_p, x_norm_p, zp, d_norm_p, stage, None, None)
        # loss_n = nn.functional.mse_loss(zn, noise)
        self.log(f'{stage}/loss_negative', loss_n, prog_bar=True, on_epoch=True)
        loss = loss_p + self.hparams.cfg.loss.weights.negative * loss_n
        return loss

    def radius_loss(self, zp, zn, margin=0., distance_factor=1.1):
        centroid = zp.mean(axis=0)
        r2p = torch.square(calculate_bounding_radius(zp, centroid)) * distance_factor
        r2n = torch.square(zn - centroid)
        loss = torch.nn.functional.relu(r2p - r2n + margin).mean()
        return loss

    def radius_step(self, batch, batch_idx, stage):
        x = batch['x']
        d = batch['d']
        mask_x = batch.get('mx', None).flatten().bool()
        mask_d = mask_x.reshape(-1,1) & mask_x.reshape(1,-1) 
        mask_d = mask_d[np.triu_indices(mask_d.size(0), k=1)]
        assert self.hparams.cfg.loss.weights.negative > 0.
        assert mask_x is not None, "Noisy negative sampling requires mask_x"
        x_norm = self.encoder.preprocessor.normalize(x)
        x_norm_p = x_norm[mask_x]
        zhat = self.encoder(x_norm)
        zp = zhat[mask_x]
        zn = zhat[~mask_x]
        # zmeans = zp.mean(axis=0)
        # zstds = zp.std(axis=0)
        # noise = torch.randn_like(zn) * zstds * 3. + zmeans
        # noise = generate_negative_samples(zp, zn.size(0))
        dp = d[mask_d]
        d_norm_p = self.encoder.preprocessor.normalize_dist(dp)
        xhat_norm_p = self.decoder(zp, unnormalize=False)
        loss_p = self.loss_function(xhat_norm_p, x_norm_p, zp, d_norm_p, stage, None, None)
        # loss_n = nn.functional.mse_loss(zn, noise)
        loss_n = self.radius_loss(zp, zn)
        self.log(f'{stage}/loss_negative', loss_n, prog_bar=True, on_epoch=True)
        loss = loss_p + self.hparams.cfg.loss.weights.negative * loss_n
        return loss
    
    def step(self, batch, batch_idx, stage):
        if self.hparams.cfg.training.mode == 'end2end':
            loss = self.end2end_step(batch, batch_idx, stage)
        elif self.hparams.cfg.training.mode == 'encoder':
            loss = self.encoder.step(batch, batch_idx, f'{stage}_encoder')
        elif self.hparams.cfg.training.mode == 'decoder':
            loss = self.decoder.step(batch, batch_idx, f'{stage}_decoder')
        elif self.hparams.cfg.training.mode == 'negative':
            loss = self.negative_step(batch, batch_idx, stage)
        elif self.hparams.cfg.training.mode == 'radius':
            loss = self.radius_step(batch, batch_idx, stage)
        else:
            raise ValueError(f"Invalid training mode: {self.hparams.cfg.training.mode}")
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'test')
        return loss
    
    def link_encoder(self): # for separate training
        self.decoder.set_encoder(self.encoder)
    
    def configure_optimizers(self):
        if self.hparams.cfg.training.mode == 'encoder':
            return torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.cfg.training.lr, weight_decay=self.hparams.cfg.training.weight_decay)
        elif self.hparams.cfg.training.mode == 'decoder':
            return torch.optim.Adam(self.decoder.parameters(), lr=self.hparams.cfg.training.lr, weight_decay=self.hparams.cfg.training.weight_decay)
        elif self.hparams.cfg.training.mode in ['end2end','negative', 'radius']:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.cfg.training.lr, weight_decay=self.hparams.cfg.training.weight_decay)
        else:
            raise ValueError(f"Invalid training mode: {self.hparams.cfg.training.mode}")

class Preprocessor(torch.nn.Module):
    def __init__(self, mean=0., std=1., dist_std=1.):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32), persistent=True)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32), persistent=True)
        self.register_buffer('dist_std', torch.tensor(dist_std, dtype=torch.float32), persistent=True)

    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def normalize_dist(self, d):
        return d / self.dist_std
    
    def unnormalize(self, x):
        return x * self.std + self.mean
    
    def unnormalize_dist(self, d):
        return d * self.dist_std
    
    def get_params(self):
        return dict(
            mean=self.mean,
            std=self.std,
            dist_std=self.dist_std
        )

class Discriminator(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        # out_dim = cfg.dimensions.get('latent')
        out_dim = 2
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)
    
    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None)
        assert mask is not None
        y = mask.flatten().long()
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def positive_proba(self, x, normalize=True):
        logits = self(x, normalize=normalize)
        return F.softmax(logits, dim=1)[:, 1]

class NoisePredictor(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        # out_dim = cfg.dimensions.get('latent')
        out_dim = 1
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)
    
    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None)
        assert mask is not None
        y = mask.flatten()
        # loss = torch.nn.functional.mse_loss(self(x), y)
        loss = nn.L1Loss()(self(x), y)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

"""
DEPRECATED. now using gradient penalty instead of clipping.
"""
class WDiscriminatorClip(Encoder):
    """
    Using weight-clipping in WGAN:
    https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L184
    """
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        # out_dim = cfg.dimensions.get('latent')
        out_dim = 1
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)
    
    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None).flatten()
        assert mask is not None
        for p in self.mlp.parameters():
            # p.data.clamp_(self.hparams.cfg.training.clamp_lower, self.hparams.cfg.training.clamp_upper)
            p.data.clamp_(- self.hparams.cfg.training.clamp, self.hparams.cfg.training.clamp)
        scores = self(x)
        mask_label = -(mask * 2 - 1.)
        loss = (scores.flatten() * mask_label).mean()

        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

"""
DEPRECATED. now using spectral normalization instead of clipping or GP.
"""
class WDGPiscriminator(Encoder):
    """
    using the gradient penalty in WGAN-GP
    https://github.com/igul222/improved_wgan_training/blob/fa66c574a54c4916d27c55441d33753dcc78f6bc/gan_toy.py#L70
    """
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        # out_dim = cfg.dimensions.get('latent')
        out_dim = 1
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)

    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None).flatten()
        assert mask is not None
        for p in self.mlp.parameters():
            # p.data.clamp_(self.hparams.cfg.training.clamp_lower, self.hparams.cfg.training.clamp_upper)
            p.data.clamp_(- self.hparams.cfg.training.clamp, self.hparams.cfg.training.clamp)
        scores = self(x)
        wgan_loss = self.compute_wgan_loss(scores, mask)
        self.log(f'{stage}/wgan_loss', wgan_loss, prog_bar=True, on_epoch=True)
        loss = self.hparams.cfg.loss.weights.wgan * wgan_loss
        
        # Compute gradient penalty conditionally
        if (batch_idx % self.hparams.cfg.training.gradient_penalty_frequency == 0):
            grad_loss = self.compute_gradient_penalty(x, mask)
            self.log(f'{stage}/grad_loss', grad_loss, prog_bar=True, on_epoch=True)
            loss += self.hparams.cfg.loss.weights.grad * grad_loss

        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def compute_wgan_loss(self, scores, mask):
        mask_label = -(mask * 2 - 1.)
        return (scores.flatten() * mask_label).mean()

    def compute_gradient_penalty(self, x, mask):
        # adapted from https://github.com/EmilienDupont/wgan-gp/blob/ef82364f2a2ec452a52fbf4a739f95039ae76fe3/training.py#L82C9-L82C66
        x_pos = x[mask.bool()]
        x_neg = x[~mask.bool()]
        sample_pts = int(min(x_pos.size(0), x_neg.size(0)) * self.hparams.cfg.training.sample_rate)
        indices_pos = torch.randperm(x_pos.size(0))[:sample_pts]
        sampled_x_pos = x_pos[indices_pos]
        indices_neg = torch.randperm(x_neg.size(0))[:sample_pts]
        sampled_x_neg = x_neg[indices_neg]
        
        alpha = torch.rand((sample_pts, 1), device=x.device, dtype=torch.float32)
        x_interp = alpha * sampled_x_pos + (1 - alpha) * sampled_x_neg
        x_interp = x_interp.clone().detach().requires_grad_(True)
        
        s_interp = self(x_interp)
        grad_outputs = torch.ones(s_interp.size(), dtype=s_interp.dtype, device=s_interp.device)
        jac = torch.autograd.grad(outputs=s_interp, inputs=x_interp, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
        jac_norm = torch.sqrt(torch.square(jac).sum(axis=1) + 1e-10)
        
        return torch.square(jac_norm - 1).mean()

    def validation_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True): # the loss requires gradient
            loss = self.step(batch, batch_idx, 'validation')
        return loss

    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True): # the loss requires gradient
            loss = self.step(batch, batch_idx, 'test')
        return loss
    
class WDiscriminator(Encoder):
    def __init__(self, cfg, preprocessor):
        super().__init__(cfg, preprocessor)
        self.preprocessor = preprocessor
        cfg.loss.dist_mse_decay = cfg.loss.get('dist_mse_decay', 0.)
        in_dim = cfg.dimensions.get('data')
        # out_dim = cfg.dimensions.get('latent')
        out_dim = 1
        self.save_hyperparameters()
        self.mlp = MLP(cfg.encoder, in_dim, out_dim)
    
    def step(self, batch, batch_idx, stage):
        x = batch['x']
        mask = batch.get('mx', None).flatten()
        assert mask is not None
        scores = self(x)
        mask_label = -(mask * 2 - 1.)
        loss = (scores.flatten() * mask_label).mean()

        self.log(f'{stage}/loss', loss, prog_bar=True, on_epoch=True)
        return loss