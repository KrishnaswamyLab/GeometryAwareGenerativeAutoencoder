"""
Train encoder and decoder separately.
By the way refactor and clean up.
"""

import torch
import pytorch_lightning as pl

activation_dict = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid()
}

class MLP(torch.nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        layer_widths = cfg.get("layer_widths", [[64, 64, 64]])
        assert len(layer_widths) >= 2, "layer_widths list must contain at least 2 elements"
        activation = cfg.get("activation", "relu")
        assert activation in activation_dict.keys(), f"activation must be one of {list(activation_dict.keys())}"
        batch_norm = cfg.get("batch_norm", False)
        dropout = cfg.get("dropout", 0.0)

        layers = []
        for i, width in enumerate(layer_widths):
            if i == 0:  # First layer, input dimension to first layer width
                layers.append(torch.nn.Linear(in_dim, width))
            else:  # Subsequent layers, previous layer width to current layer width
                layers.append(torch.nn.Linear(layer_widths[i-1], width))

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(width))
            activation_func = activation_dict[activation]
            layers.append(activation_func)

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            
        layers.append(torch.nn.Linear(layer_widths[-1], out_dim))
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
        zhat = self.encoder(x)
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
    
    def step(self, batch, batch_idx, stage):
        if self.hparams.cfg.training.mode == 'end2end':
            loss = self.end2end_step(batch, batch_idx, stage)
        elif self.hparams.cfg.training.mode == 'encoder':
            loss = self.encoder.step(batch, batch_idx, f'{stage}_encoder')
        elif self.hparams.cfg.training.mode == 'decoder':
            loss = self.decoder.step(batch, batch_idx, f'{stage}_decoder')
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
        elif self.hparams.cfg.training.mode == 'end2end':
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