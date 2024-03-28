import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from unified_model import GeometricAE
sys.path.append('../../src/')
from data import RowStochasticDataset
from model import AEProb, Decoder

from utils.log_utils import log
from utils.seed import seed_everything
from utils.early_stop import EarlyStopping

activation_dict = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid()
}

class AffinityMatching(GeometricAE):
    ''' Geometric Autoencoder with affinity matching encoding.'''
    def __init__(self, 
                 ambient_dimension, 
                 latent_dimension, 
                 loss_type, # 'kl', 'jsd', 'm-divergence',
                 model_type = 'affinity', 
                 activation = 'relu',
                 layer_widths = [256, 128, 64],
                 kernel_method = 'gaussian',
                 kernel_alpha = 1,
                 kernel_bandwidth = 1,
                 knn = 5, # Phate KNN
                 t = 0, # Phate t, 0 is auto
                 n_landmark = 5000, # Phate n_landmark
                 verbose = False):
        super().__init__(ambient_dimension, latent_dimension, model_type)
        self.loss_type = loss_type
        
        self.activation = activation
        self.layer_widths = layer_widths
        
        self.kernel_method = kernel_method
        self.kernel_alpha = kernel_alpha
        self.kernel_bandwidth = kernel_bandwidth

        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        
        self.verbose = verbose

        self.encoder = None
        self.decoder = None
    
    def fit(self, 
            X, 
            train_mask,
            percent_test,
            data_name,
            max_epochs,
            batch_size,
            lr,
            shuffle,
            weight_decay,
            monitor='val_loss',
            patience=100,
            seed=2024,
            log_every_n_steps=100,
            accelerator='auto',
            train_from_scratch=True,
            model_save_path='./affinity_matching', # if None, train from scratch; else load from model_save_path
            ):
        
        seed_everything(seed)

        if train_mask is None:
            # Generate train_mask
            idxs = np.random.permutation(X.shape[0])
            split_idx = int(X.shape[0] * (1-percent_test))
            train_mask = np.zeros(X.shape[0], dtype=int)
            train_mask[idxs[:split_idx]] = 1
            train_mask = train_mask.astype(bool)
        
        train_val_data = X[train_mask]
        split_val_idx = int(len(train_val_data)*(1-percent_test))
        train_data = train_val_data[:split_val_idx]
        val_data = train_val_data[split_val_idx:]
        test_data = X[~train_mask]

        train_dataset = RowStochasticDataset(data_name=data_name, X=train_data, X_labels=None, dist_type='phate_prob', 
                                             knn=self.knn, t=self.t, n_landmark=self.n_landmark)
        train_val_dataset = RowStochasticDataset(data_name=data_name, X=train_val_data, X_labels=None, dist_type='phate_prob', 
                                                 knn=self.knn, t=self.t, n_landmark=self.n_landmark)
        whole_dataset = RowStochasticDataset(data_name=data_name, X=X, X_labels=None, dist_type='phate_prob',
                                                knn=self.knn, t=self.t, n_landmark=self.n_landmark)
        log(f'Train dataset: {len(train_dataset)}; \
          Val dataset: {len(train_val_dataset)}; \
          Whole dataset: {len(whole_dataset)}')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False)
        

        ''' Fit the model to the data X. '''
        act_fn = activation_dict[self.activation]
        encoder = AEProb(dim=self.ambient_dimension, emb_dim=self.latent_dimension, 
                        layer_widths=self.layer_widths, activation_fn=act_fn,
                        prob_method=self.kernel_method, dist_reconstr_weights=[1.0, 0.0])
        decoder = Decoder(dim=self.ambient_dimension, emb_dim=self.latent_dimension,
                          layer_widths=self.layer_widths[::-1], activation_fn=act_fn)
        
        if train_from_scratch is False:
            encoder.load_from_checkpoint(os.path.join(model_save_path, 'encoder.ckpt'))
            log(f'Loaded encoder from {model_save_path}, skipping encoder training ...')

            decoder.load_from_checkpoint(os.path.join(model_save_path, 'decoder.ckpt'))
            log(f'Loaded decoder from {model_save_path}, skipping decoder training ...')
        else:
            os.makedirs(model_save_path, exist_ok=True)

            self._train_encoder(encoder, 
                                train_dataset, train_loader, train_val_dataset, train_val_loader,
                                max_epochs, lr, weight_decay, patience, log_every_n_steps, accelerator,
                                os.path.join(model_save_path, 'encoder.ckpt'), wandb_run=None)
            
            # Use embeddings from encoder to train decoder. Keep encoder frozen.
            self._train_decoder(encoder, decoder, 
                                train_data, val_data, test_data,
                                max_epochs, batch_size, lr, weight_decay, patience, log_every_n_steps, accelerator,
                                os.path.join(model_save_path, 'decoder.ckpt'), wandb_run=None)
            

            encoder.load_from_checkpoint(os.path.join(model_save_path, 'encoder.ckpt'))
            decoder.load_from_checkpoint(os.path.join(model_save_path, 'decoder.ckpt'))
        
        self.encoder = encoder
        self.decoder = decoder

        print('Done fitting model.')
        

    def _train_encoder(self, encoder, 
                       train_dataset, train_loader, train_val_dataset, train_val_loader, 
                       max_epochs, lr, weight_decay, patience, log_every_n_steps, accelerator,
                       model_save_path, wandb_run=None):
        
        device_av = "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator is None or accelerator == 'auto':
            device = device_av
        else:
            device = accelerator

        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopper = EarlyStopping(mode='min',
                                    patience=patience,
                                    percentage=False)

        best_metric = np.inf
        encoder = encoder.to(device)

        log('Training Encoder ...')
        for eid in range(max_epochs):
            encoder.train()

            encoder_loss = 0.0
            train_Z = []
            optimizer.zero_grad()

            for (_, batch_x) in train_loader:
                batch_x = batch_x.to(device)

                batch_z = encoder.encode(batch_x)
                train_Z.append(batch_z)
            
            # row-wise prob divergence loss
            train_Z = torch.cat(train_Z, dim=0) #[N, emb_dim]

            pred_prob_matrix = encoder.compute_prob_matrix(train_Z, 
                                                        t=train_dataset.t, 
                                                        alpha=self.kernel_alpha, 
                                                        bandwidth=self.kernel_bandwidth)
            gt_prob_matrix = (train_dataset.row_stochastic_matrix).type(torch.float32).to(device)
            encoder_loss = encoder.encoder_loss(gt_prob_matrix, pred_prob_matrix, type=self.loss_type)
        
            encoder_loss.backward()
            optimizer.step()

            if eid == 0 or eid % log_every_n_steps == 0:
                log(f'[Epoch: {eid}]: Encoder Loss: {encoder_loss.item()}')
                if wandb_run is not None:
                    wandb_run.log({'train/encoder_loss': encoder_loss.item()})

            ''' Validation (used both train + val data to compute prob matrix)'''
            encoder.eval()
            val_encoder_loss = 0.0
            val_Z = []
            with torch.no_grad():
                for (_, batch_x) in train_val_loader:
                    batch_x = batch_x.to(device)

                    batch_z = encoder.encode(batch_x)
                    val_Z.append(batch_z)
                
                val_Z = torch.cat(val_Z, dim=0)

                train_val_pred_prob_matrix = encoder.compute_prob_matrix(val_Z, t=train_val_dataset.t, 
                                                                    alpha=self.kernel_alpha, 
                                                                    bandwidth=self.kernel_bandwidth)
                gt_train_val_prob_matrix = (train_val_dataset.row_stochastic_matrix).type(torch.float32).to(device)
                val_encoder_loss = encoder.encoder_loss(gt_train_val_prob_matrix, 
                                                    train_val_pred_prob_matrix,
                                                    type=self.loss_type)
                
                log(f'\n[Epoch: {eid}]: Val Encoder Loss: {val_encoder_loss.item()}')
                if wandb_run is not None:
                    wandb_run.log({'val/encoder_loss': val_encoder_loss.item()})

            if val_encoder_loss < best_metric:
                log('\nBetter model found. Saving best model ...\n')
                best_metric = val_encoder_loss
                encoder.save_weights(model_save_path)

            # Early Stopping
            if early_stopper.step(val_encoder_loss):
                log('[Encoder] Early stopping criterion met. Ending training.\n')
                break

        log('Done training encoder.')
    

    def _train_decoder(self, encoder, decoder, 
                       train_data, val_data, test_data,
                       max_epochs, batch_size, lr, weight_decay, patience, log_every_n_steps, accelerator,
                       model_save_path, wandb_run=None):
        
        # Use embeddings from encoder to train decoder. Keep encoder frozen.
        encoder.eval()

        train_data = torch.tensor(train_data, dtype=torch.float32)
        val_data = torch.tensor(val_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        with torch.no_grad():
            train_Z = encoder.encode(train_data)
            val_Z = encoder.encode(val_data)
            test_Z = encoder.encode(test_data)
        
        # dataset for decoder: z, x
        train_decoder_dataset = TensorDataset(train_Z, train_data)
        val_decoder_dataset = TensorDataset(val_Z, val_data)
        test_decoder_dataset = TensorDataset(test_Z, test_data)

        train_loader = DataLoader(train_decoder_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_decoder_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_decoder_dataset, batch_size=batch_size, shuffle=False)


        device_av = "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator is None or accelerator == 'auto':
            device = device_av
        else:
            device = accelerator

        optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopper = EarlyStopping(mode='min',
                                    patience=patience,
                                    percentage=False)

        best_metric = np.inf
        decoder = decoder.to(device)

        for eid in range(max_epochs):
            decoder.train()
            decoder_epoch_loss = 0.0

            for _, (batch_z, batch_x) in enumerate(train_loader):
                batch_z = batch_z.to(device)
                batch_x = batch_x.to(device)

                optimizer.zero_grad()


                batch_x_hat = decoder(batch_z)
                decoder_loss = torch.nn.functional.mse_loss(batch_x, batch_x_hat)
                decoder_epoch_loss += decoder_loss.item()

                decoder_loss.backward()
                optimizer.step()

            decoder_epoch_loss /= len(train_loader)
            if eid % log_every_n_steps == 0:
                log(f'[Epoch: {eid}]: Decoder Loss: {decoder_epoch_loss}')
                if wandb_run is not None:
                    wandb_run.log({'train/decoder_loss': decoder_epoch_loss})

            # Validation decoder
            decoder.eval()
            val_decoder_loss = 0.0
            with torch.no_grad():
                for _, (batch_z, batch_x) in enumerate(val_loader):
                    batch_z = batch_z.to(device)
                    batch_x = batch_x.to(device)

                    batch_x_hat = decoder(batch_z)
                    val_decoder_loss += torch.nn.functional.mse_loss(batch_x, batch_x_hat).item()

                val_decoder_loss /= len(val_loader)
                log(f'[Epoch: {eid}]: Val Decoder Loss: {val_decoder_loss}', to_console=True)
                if wandb_run is not None:
                    wandb_run.log({'val/decoder_loss': val_decoder_loss})

            if val_decoder_loss < best_metric:
                log('Better model found. Saving best model ...\n')
                best_metric = val_decoder_loss
                decoder.save_weights(model_save_path)

            # Early Stopping
            if early_stopper.step(val_decoder_loss):
                log('[Decoder] Early stopping criterion met. Ending training.\n')
                break

            log('Done training decoder.')
    
    def encode(self, X):
        ''' Encode input data X to latent space. '''
        if self.encoder is None:
            raise ValueError('Encoder not trained yet. Please train the model first.')
        
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            Z = self.encoder.encode(X)
        
        return Z
    
    def decode(self, Z):
        ''' Decode latent space Z to ambient space. '''
        if self.decoder is None:
            raise ValueError('Decoder not trained yet. Please train the model first.')
        
        with torch.no_grad():
            X_hat = self.decoder(Z)
        
        return X_hat
    


if __name__ == "__main__":
    model_hypers = {
        'ambient_dimension': 10,
        'latent_dimension': 2,
        'model_type': 'affinity',
        'loss_type': 'kl',
        'activation': 'relu',
        'layer_widths': [256, 128, 64],
        'kernel_method': 'gaussian',
        'kernel_alpha': 1,
        'kernel_bandwidth': 1,
        'knn': 5,
        't': 0,
        'n_landmark': 5000,
        'verbose': False
    }
    training_hypers = {
        'data_name': 'randomtest',
        'max_epochs': 100,
        'batch_size': 64,
        'lr': 1e-3,
        'shuffle': True,
        'weight_decay': 1e-5,
        'monitor': 'val_loss',
        'patience': 100,
        'seed': 2024,
        'log_every_n_steps': 100,
        'accelerator': 'auto',
        'train_from_scratch': False,
        'model_save_path': './affinity_matching'
    }
    # Test AffinityMatching model
    X = np.random.randn(1000, 10) # 3000 samples, 10 features
    model = AffinityMatching(**model_hypers)
    model.fit(X, train_mask=None, percent_test=0.3, **training_hypers)

    Z = model.encode(X)
    print('Encoded Z:', Z.shape)
    X_hat = model.decode(Z)
    print('Decoded X:', X_hat.shape)
