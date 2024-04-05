import torch
import numpy as np

import phate
from sklearn.manifold import TSNE
import umap
from other_methods import DiffusionMap

class GeometricAE:
    def __init__(
        self,
        ambient_dimension,
        latent_dimension,
        model_type, # "distance" or "affinity"
        # extra hyperparameters, with sane defaults
    ):
        self.ambient_dimension = ambient_dimension
        self.latent_dimension = latent_dimension
        self.model_type = model_type
        # match self.model_type:
        #     case "distance":
        #         self.model = 'get distance model here' # initialize with hyperparameters
        #     case 'affinity':
        #         self.model = 'get affinity model here'
        #     case _:
        #         raise NotImplementedError("Invalid Model Type")
        if self.model_type == "distance":
            self.model = 'get distance model here'
        elif self.model_type == "affinity":
            self.model = 'get affinity model here'
        else:
            raise NotImplementedError("Invalid Model Type")

    def fit(
        self,
        X, # pointcloud with assumed local euclidean distances
        train_mask = None, # bool, mask for training points
        percent_test = 0.3, # train/test split, if train_mask is not provided
        **kwargs, # other hyperparams of graph creation, including default phate
    ):
        # Compute PHATE distances/affinities
        # Create pytorch PointCloud dataset, tailored to the model, with given train test split. 
        # training loop
        pass
        
    def fit_transform(self, 
                      X, 
                      X_test,
                      n_epochs):
        self.fit()
        return self.encode(X)
    
    def evaluate(self,
                 data_path,
                 **kwargs):
        '''
        Fit the model on the data path & Evaluate both encoder and decoder.
        **kwargs: extra hyperparameters for fitting the model.
        '''

        data = np.load(data_path, allow_pickle=True)
        true_data = data['data_gt']
        raw_data = data['data']
        labels = data['colors']
        train_mask = data['is_train']
        if 'int' in train_mask.dtype.name:
            train_mask = train_mask.astype(bool)
        
        # Fit the model
        self.fit(raw_data, train_mask, **kwargs)

        assert self.encoder is not None, "Encoder not fit"
        assert self.decoder is not None, "Decoder not fit"

        # Encode the data
        pred_embed = self.encode(raw_data).cpu().detach().numpy()
        # Other embeddings for comparison
        phate_embed = phate.PHATE(
            n_components=self.ambient_dimension, 
            k=self.knn,
            t=self.t if self.t != 0 else 'auto',
            n_landmark=self.n_landmark,
        ).fit_transform(raw_data)
        tsne_embed = TSNE(n_components=self.ambient_dimension, perplexity=5).fit_transform(raw_data)
        umap_embed = umap.UMAP().fit_transform(raw_data)
        dm_embed = DiffusionMap().fit_transform(raw_data)

        # TODO: Evaluate the embeddings
        ''' DeMAP '''

        
    def encode(self, X):
        # Call the encoder function of the model
        pass

    def decode(self, Z):
        pass 

    def encoder_pullback(self, x):
        '''
        Pullback the metric from the latent space (n) to the input space (D).
        J = df/dx (B, n, D)
        metric = J^T J
        Inputs:
            x: [B, D]
        Returns:
            metric: [B, D, D] metric tensor
        '''
        x.requires_grad = True
        J = torch.autograd.functional.jacobian(self.encode, x, create_graph=True) # [B, n, B, D]

        J = torch.stack([J[i, :, i, :] for i in range(J.shape[0])]) # [B, n, D]

        pullback_metric = torch.matmul(J.transpose(1, 2), J) # [B, D, D]
        
        return pullback_metric
    
        
    
        
    
        
    


    

    


	