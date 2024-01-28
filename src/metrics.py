'''
Various metrics for evaluating the performance of modeling the data manifold.
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors

def distance_distortion(pred_distances: np.ndarray, 
                        gt_distances: np.ndarray) -> float:
    '''
    pred_distances: ndarray (N, N), predicted distance matrix
    gt_distances: ndarray (N, N), ground truth distance matrix
    '''
    gt_distances = gt_distances.astype(np.float32)
    pred_distances = pred_distances.astype(np.float32)

    # Remove the diagonal
    gt_distances = gt_distances[~np.eye(gt_distances.shape[0],dtype=bool)].reshape(gt_distances.shape[0],-1)
    pred_distances = pred_distances[~np.eye(pred_distances.shape[0],dtype=bool)].reshape(pred_distances.shape[0],-1)

    # compute the distortion
    eps = 1e-8
    distortion = np.mean(np.abs(gt_distances - pred_distances) / (gt_distances + eps))

    return float(distortion)


def mAP(embeddings: np.ndarray, 
        input: np.ndarray,
        graph_adjacency: np.ndarray,
        distance_op: str = 'norm') -> float:
    '''
    embeddings: ndarray (N, embedding_dim)
    input: ndarray (N, input_dim)
    graph_adjacency: ndarray (N, N)
    distance_op: str, 'norm'|'dot'|'cosine'
    
    AP_(xi, xj) := \frac{the number of neighbors of xi, \
    enclosed by smallest ball that contains xj centered at xi}{the points enclosed by the ball centered at xi}

    graph_adjacency[i, j] = 1 if i and j are neighbors, 0 otherwise

    Returns:
        mAP: float
    '''
    N = embeddings.shape[0]
    assert N == input.shape[0] == graph_adjacency.shape[0] == graph_adjacency.shape[1]

    # compute the distance matrix
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if distance_op == 'norm':
                distance_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
            elif distance_op == 'dot':
                distance_matrix[i, j] = np.dot(embeddings[i], embeddings[j])
            elif distance_op == 'cosine':
                distance_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / \
                                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            else:
                raise Exception('distance_op must be either norm or dot')
    
    # compute the AP
    AP = np.zeros(N)
    for i in range(N):
        # find the neighbors of i
        neighbors = np.argwhere(graph_adjacency[i] == 1).flatten()
        # compute the distance between i and its neighbors
        distances = distance_matrix[i, neighbors] # (n_neighbors, )
        for j in range(len(neighbors)):
            # compute the number of points enclosed by the ball_j centered at i
            all_enclosed = np.argwhere(distance_matrix[i] <= distances[j]).flatten() 
            # compute the number of neighbors of enclosed by the ball_j centered at i
            n_enclosed_j = len(np.intersect1d(all_enclosed, neighbors))
            # compute the AP
            if n_enclosed_j > 0:
                AP[i] += n_enclosed_j / all_enclosed.shape[0]
        
        if len(neighbors) > 0:
            AP[i] /= len(neighbors)
    
    mAP = np.mean(AP)

    return mAP

def computeKNNmAP(embeddings: np.ndarray, 
        input: np.ndarray,
        k: int,
        distance_op: str = 'norm') -> float:
    # Create the k-NN model
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(input)

    # Find k-nearest neighbors for each point in X
    distances, indices = knn.kneighbors(input)

    # Initialize the affinity matrix
    n_samples = input.shape[0]
    affinity_matrix = np.zeros((n_samples, n_samples))

    # Update the affinity matrix
    for i in range(n_samples):
        for j in indices[i]:
            if i != j:  # Exclude self-loops
                affinity_matrix[i, j] = 1
    mapscore = mAP(embeddings, input, affinity_matrix, distance_op)
    return mapscore