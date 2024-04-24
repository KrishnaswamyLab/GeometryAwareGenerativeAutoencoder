import numpy as np
from diffusionmap import DiffusionMap

def add_negative_samples(data_dict, subset_rate=0.5, noise_rate=1, seed=42, noise='gaussian', mask_dists=False):
    np.random.seed(seed)
    data_train = data_dict['data'][data_dict['is_train']]
    subset_id = np.random.choice(data_train.shape[0], int(data_train.shape[0] * subset_rate), replace=True)

    if noise == 'gaussian':
        data_subset = data_train[subset_id]
        data_std = data_subset.std()
        noise = np.random.normal(0, data_std * noise_rate, data_subset.shape)
        data_subset_noisy = data_subset + noise
    elif noise == 'hi-freq':
        diff_map_op = DiffusionMap(n_components=3, t=3, random_state=seed).fit(data_train)
        data_subset_noisy = data_train + make_hi_freq_noise(data_train, diff_map_op, noise_rate=noise_rate, add_data_mean=False)
        data_subset_noisy = data_subset_noisy[subset_id]
    elif noise == 'hi-freq-no-add':
        diff_map_op = DiffusionMap(n_components=3, t=3, random_state=seed).fit(data_train)
        data_subset_noisy = make_hi_freq_noise(data_train, diff_map_op, noise_rate=noise_rate, add_data_mean=True)
        data_subset_noisy = data_subset_noisy[subset_id]
    else:
        raise ValueError(f"Unknown noise type: {noise}")
    dists = data_dict['dist']
    dists_B = np.ones((dists.shape[0], data_subset_noisy.shape[0])) * dists.std() * 3
    dists_D = np.zeros((data_subset_noisy.shape[0], data_subset_noisy.shape[0]))
    dists_new = np.r_[np.c_[dists, dists_B], np.c_[dists_B.T, dists_D]]

    mask_d_A = np.ones(dists.shape, dtype=np.float32)
    mask_d_B = np.ones((dists.shape[0], data_subset_noisy.shape[0]), dtype=np.float32)
    # FIXME should be zeros? need to train and test! if zeros, then they can be anywhere.
    if mask_dists:
        mask_d_D = np.zeros((data_subset_noisy.shape[0], data_subset_noisy.shape[0]), dtype=np.float32)
    else:
        mask_d_D = np.ones((data_subset_noisy.shape[0], data_subset_noisy.shape[0]), dtype=np.float32)
    mask_d = np.r_[np.c_[mask_d_A, mask_d_B], np.c_[mask_d_B.T, mask_d_D]]

    x = data_dict['data']
    x_new = np.r_[x, data_subset_noisy]
    mask_x = np.r_[np.ones(x.shape[0]), np.zeros(data_subset_noisy.shape[0])]

    # phate_coords = data_dict['phate']
    # phate_coords_new = np.r_[phate_coords, np.zeros((data_subset_noisy.shape[0], phate_coords.shape[1]))]

    colors = data_dict['colors']
    colors_new = np.r_[colors, np.zeros((data_subset_noisy.shape[0]))]

    train_mask = data_dict['is_train']
    train_mask_new = np.r_[train_mask, np.ones(data_subset_noisy.shape[0])]

    data_dict['data'] = x_new
    data_dict['dist'] = dists_new
    data_dict['colors'] = colors_new
    data_dict['is_train'] = train_mask_new
    data_dict['mask_x'] = mask_x
    data_dict['mask_d'] = mask_d

    return data_dict 

def make_hi_freq_noise(dataX, diff_map_op, noise_rate=0.1, seed=42, add_data_mean=False):
    np.random.seed(seed)
    white_noise = np.random.randn(*dataX.shape)
    noise_filter = diff_map_op.Dinvhf @ diff_map_op.eigenvectors * (1 - diff_map_op.eigenvalues ** diff_map_op.t)
    noise_noise = noise_filter @ white_noise
    noise_noise = noise_noise - noise_noise.mean(axis=0)
    noise_noise = noise_noise / noise_noise.std(axis=0) * dataX.std(axis=0) * noise_rate
    if add_data_mean:
        noise_noise = noise_noise + dataX.mean(axis=0)
    return noise_noise
    

# def add_negative_samples(data_dict, subset_rate=0.5, noise_rate=0.2, seed=42):
#     np.random.seed(seed)
#     data_train = data_dict['data'][data_dict['is_train']]

#     subset_id = np.random.choice(data_train.shape[0], int(data_train.shape[0] * subset_rate), replace=True)
#     data_subset = data_train[subset_id]
#     data_std = data_subset.std()
#     noise = np.random.normal(0, data_std * noise_rate, data_subset.shape)
#     data_subset_noisy = data_subset + noise

#     dists = data_dict['dist']
#     dists_B = np.ones((dists.shape[0], data_subset_noisy.shape[0])) * dists.std() * 3
#     dists_D = np.zeros((data_subset_noisy.shape[0], data_subset_noisy.shape[0]))
#     dists_new = np.r_[np.c_[dists, dists_B], np.c_[dists_B.T, dists_D]]

#     mask_d_A = np.ones(dists.shape, dtype=np.float32)
#     mask_d_B = np.ones((dists.shape[0], data_subset_noisy.shape[0]), dtype=np.float32)
#     mask_d_D = np.ones((data_subset_noisy.shape[0], data_subset_noisy.shape[0]), dtype=np.float32)
#     mask_d = np.r_[np.c_[mask_d_A, mask_d_B], np.c_[mask_d_B.T, mask_d_D]]

#     x = data_dict['data']
#     x_new = np.r_[x, data_subset_noisy]
#     mask_x = np.r_[np.ones(x.shape[0]), np.zeros(data_subset_noisy.shape[0])]

#     # phate_coords = data_dict['phate']
#     # phate_coords_new = np.r_[phate_coords, np.zeros((data_subset_noisy.shape[0], phate_coords.shape[1]))]

#     colors = data_dict['colors']
#     colors_new = np.r_[colors, np.zeros((data_subset_noisy.shape[0]))]

#     train_mask = data_dict['is_train']
#     train_mask_new = np.r_[train_mask, np.ones(data_subset_noisy.shape[0])]

#     data_dict['data'] = x_new
#     data_dict['dist'] = dists_new
#     data_dict['colors'] = colors_new
#     data_dict['is_train'] = train_mask_new
#     data_dict['mask_x'] = mask_x
#     data_dict['mask_d'] = mask_d

#     return data_dict