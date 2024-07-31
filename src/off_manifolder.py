"""
from https://github.com/professorwug/autometric/blob/2412feadf8fde7328c1e830709b85aa853c224e4/nbs/library/off-manifold-pullback.ipynb
"""
import torch
class OffManifolderLinear():
    """
    Folds points off manifold into higher dimensions using random matrices.
    """
    def __init__(self,
                 X, # n x d points sampled from manifold (in latent space)
                 density_loss_function = None, # function that takes a batch of tensors as input, and outputs a scalar which is 0 on the manifold, and bigger further away.
                 folding_dim = 10,
                 density_k = 5,
                 density_tol = 0.1,
                 density_exponential = 4, 
                 # modify to pass in density_loss
                ):
        self.X = X
        self.device = X.device
        self.dim = X.shape[1]
        self.folding_dim = folding_dim
        self.density_k = density_k
        self.density_tol = density_tol
        self.density_exponential = density_exponential
        self.density_loss_function = density_loss_function
        
        self.preserve_matrix = torch.zeros(self.dim, self.folding_dim, dtype=torch.float).to(self.device)
        for i in range(self.dim):
            self.preserve_matrix[i,i] = 1.0

        self.random_matrix = torch.randn(self.dim, self.folding_dim).to(self.device)
        self.random_matrix[:self.dim, :self.dim] = torch.zeros(self.dim, self.dim).to(self.device)
        # self.random_layer = torch.nn.Linear(self.dim, self.folding_dim)

    def _1density_loss(self, a):
        # 0 for points on manifold within tolerance. Designed for a single point.
        dists = torch.linalg.norm(self.X - a, axis=1)
        print(dists.shape)
        smallest_k_dists, idxs = torch.topk(dists, self.density_k, largest=False) # return k smallest distances
        loss = torch.sum(
            torch.nn.functional.relu( smallest_k_dists - self.density_tol )
        )
        return loss
    def density_loss(self, points):
        if self.density_loss_function is not None:
            return self.density_loss_function(points)
        else:
            return torch.vmap(self._1density_loss)(points)

    def immersion(self, points):
        preserved_subspace = points @ self.preserve_matrix
        random_dirs = points @ self.random_matrix
        # random_dirs = self.random_layer(points)
        weighting_factor = torch.exp(self.density_loss(points)*self.density_exponential) - 1 # starts at 1; gets higher immediately.
        print(f"{preserved_subspace.shape} {random_dirs.shape} {weighting_factor.shape}")
        return preserved_subspace + random_dirs*weighting_factor[:,None]

    def pullback_metric(self, points):
        if not isinstance(points, torch.Tensor): points = torch.tensor(points, dtype=torch.float)
        jac = torch.func.jacrev(self.immersion, argnums = 0) #(points)
        def pullback_per_point(p):
            print(p)
            print(p.shape)
            J = jac(p[None,:])
            J = torch.squeeze(J)
            print("shape J", J.shape)
            return J.T @ J
        return torch.vmap(pullback_per_point)(points)

# def offmanifolder_maker(
#     encoder, 
#     discriminator, 
#     emb_dim,
#     device,
#     folding_dim = 10,
#     density_exponential = 4,
# ):
#     assert folding_dim > emb_dim
#     # preserve_matrix = torch.zeros(emb_dim, folding_dim, dtype=torch.float).to(device)
#     # random_matrix = torch.randn(1, folding_dim - emb_dim).to(device)
#     random_matrix = torch.ones(1, folding_dim - emb_dim, dtype=torch.float, device=device)
#     def ofm_(x):
#         z = encoder(x)
#         # preserved_subspace = z @ preserve_matrix
#         preserved_subspace = torch.cat([z, torch.zeros((z.size(0), folding_dim-emb_dim), dtype=z.dtype, device=device)], dim=1)
#         random_dirs = torch.cat([torch.zeros((z.size(0), emb_dim), dtype=z.dtype, device=device), random_matrix.repeat(z.size(0), 1)], dim=1)
#         # weighting_factor = torch.exp((1-discriminator(x))*density_exponential) - 1 # starts at 1; gets higher immediately.
#         weighting_factor = ((1-discriminator(x))*density_exponential) # starts at 1; gets higher immediately.
#         return preserved_subspace + random_dirs*weighting_factor[:,None]
#     return ofm_

#     # large dim / rand prj moves the pt far away

# def offmanifolder1_maker(
#     encoder_, 
#     discriminator_, 
#     disc_factor = 4,
# ):
#     def ofm_(x):
#         z = encoder_(x)
#         weighting_factor = discriminator_(x)*disc_factor
#         return torch.cat([z, weighting_factor.unsqueeze(1).repeat(1, 1)], dim=1)
#     return ofm_

# def offmanifolder2_maker(
#     encoder_, 
#     discriminator_, 
#     disc_factor = 4,
# ):
#     def ofm_(x):
#         z = encoder_(x)
#         # weighting_factor = discriminator_(x)*disc_factor
#         weighting_factor = torch.exp((1-discriminator_(x))*disc_factor) - 1
#         return torch.cat([z, weighting_factor.unsqueeze(1).repeat(1, 1)], dim=1)
#     return ofm_

# def offmanifolder3_maker(
#     encoder_, 
#     discriminator_, 
#     disc_factor = 4,
#     folding_dim = 10,
# ):
#     def ofm_(x):
#         z = encoder_(x)
#         # weighting_factor = discriminator_(x)*disc_factor
#         weighting_factor = torch.exp((1-discriminator_(x))*disc_factor) - 1
#         rand_pts = torch.rand(z.shape[0], folding_dim - z.shape[1], device=x.device, dtype=torch.float32)
#         return torch.cat([z, weighting_factor.reshape(-1,1) * rand_pts], dim=1)
#     return ofm_

# def offmanifolder4_maker(
#     encoder_, 
#     discriminator_,
#     device,
#     disc_factor = 4,
#     emb_dim=2,
#     folding_dim = 10,
# ):
#     random_matrix = torch.randn(emb_dim, folding_dim - emb_dim, device=device, dtype=torch.float32)
#     def ofm_(x):
#         z = encoder_(x)
#         rand_pts = z @ random_matrix
#         # weighting_factor = discriminator_(x)*disc_factor
#         weighting_factor = torch.exp((1-discriminator_(x))*disc_factor) - 1
#         return torch.cat([z, weighting_factor.reshape(-1,1) * rand_pts], dim=1)
#     return ofm_

def offmanifolder5_maker(
    encoder_, 
    discriminator_, 
    disc_factor = 4,
    max_prob=1.
):
    def ofm_(x):
        z = encoder_(x)
        weighting_factor1 = (max_prob-discriminator_(x))/max_prob*disc_factor
        weighting_factor2 = torch.exp(weighting_factor1)
        return torch.cat([z * (weighting_factor2.reshape(-1,1)), (weighting_factor1 * 100).unsqueeze(1).repeat(1, 1)], dim=1)
    return ofm_

offmanifolder_maker = offmanifolder5_maker


def offmanifolder6_maker(
    encoder_, 
    discriminator_, 
    disc_factor = 4,
    # max_prob=1.
):
    def ofm_(x):
        z = encoder_(x)
        # weighting_factor1 = (max_prob-discriminator_(x))/max_prob*disc_factor
        # weighting_factor2 = torch.exp(weighting_factor1)
        return torch.cat([z, (discriminator_(x) * disc_factor).unsqueeze(1).repeat(1, 1)], dim=1)
    return ofm_

offmanifolder_maker_new = offmanifolder6_maker