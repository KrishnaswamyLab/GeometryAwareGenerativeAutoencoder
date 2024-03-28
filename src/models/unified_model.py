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
                 X,
                 X_test_idx):
        pass
        
    def encode(self, X):
        # Call the encoder function of the model
        pass

    def decode(self, Z):
        pass 
        
    
        
    
        
    
        
    


    

    


	