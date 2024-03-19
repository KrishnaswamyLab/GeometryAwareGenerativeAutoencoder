class GeometricAE:
    def __init__(
        self,
        ambient_dimension,
        latent_dimension,
        model_type, # "distance" or "affinity"
        # extra hyperparameters
    ):
        self.model = None # Get model
        # do stuff
        pass

    def fit(
        self,
        X, # pointcloud with assumed local euclidean distances
        percent_test = 0.3, # train/test split
        n_epochs = 100, # other hyperparams of graph creation, including default phate
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
        
    def encode(self):
        # Call the encoder function of the model
        pass

    def decode(self):
        pass 
        
    
        
    
        
    
        
    


    

    


	