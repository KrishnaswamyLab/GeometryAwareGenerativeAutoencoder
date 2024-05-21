import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
sys.path.append('../../src/')
# from diffusion import DiffusionModel
# from evaluate import get_results
from omegaconf import OmegaConf
# from main import load_data, make_model
import numpy as np
import os
import glob
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import torch
from model2 import Autoencoder, Preprocessor, WDiscriminator
from off_manifolder import offmanifolder_maker
import magic
import torch
import pathlib
import copy

import wandb
import sys
import matplotlib.pyplot as plt
import scprep
import pandas as pd
from omegaconf import OmegaConf
import os
import glob
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from geodesic import jacobian, velocity, CondCurve, GeodesicBridgeOverfit, GeodesicBridge
from plotly3d.plot import scatter, trajectories
import torch
from torch import nn
import pytorch_lightning as pl
from procrustes import Procrustes
from torch.utils.data import DataLoader, TensorDataset


import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(cfg_main: DictConfig):
    print(OmegaConf.to_yaml(cfg_main))  # This prints the entire configuration to the console
    
    if cfg_main.use_wandb:
        config = OmegaConf.to_container(cfg_main, resolve=True, throw_on_missing=True)
        run_main = wandb.init(
            entity="xingzhis",
            project="dmae",
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )

    results_path = f'results/{cfg_main.data_name}/{cfg_main.dimensions_latent}'
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

# main_sweep_id = '59muwbsf'
# main_disc_sweep_id = 'w8hu793l'
# main_data_name = 'saddle_none_0'
# main_weights_cycle = 1.
# main_dimensions_latent = 3

    wandb.login()
    api = wandb.Api()

    entity = "xingzhis"
    project = "dmae"
    sweep_id = cfg_main.ae_sweep_id
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    # Initialize an empty list to store run data
    runs_data = []


    # Iterate through each run in the sweep
    for run in sweep.runs:
        # Extract metrics and configs
        metrics = run.summary._json_dict
        configs = run.config
        
        # Combine metrics and configs, and add run ID
        combined_data = {**metrics, **configs, "run_id": run.id}
        
        # Append the combined data to the list
        runs_data.append(combined_data)

    # Create a DataFrame from the runs data
    df = pd.DataFrame(runs_data)

    data_name = cfg_main.data_name
    # run_ids = df[(df['data.name'] == data_name) & (df['cfg/loss/weights/cycle'] == 1.) & (df['cfg/dimensions/latent'] == 3)]['run_id']
    run_ids = df[(df['data.name'] == data_name) & (df['loss.weights.cycle'] == cfg_main.weights_cycle) & (df['dimensions.latent'] == cfg_main.dimensions_latent)]['run_id']
    assert len(run_ids) == 1
    run_id = run_ids.iloc[0]
    run = api.run(f"{entity}/{project}/{run_id}")
    # run = api.run(f"{entity}/{project}/{run_id}")
    folder_path = '../../src/wandb/'
    cfg = OmegaConf.create(run.config)
    folder_list = glob.glob(f"{folder_path}*{run.id}*")
    ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
    ckpt_path = ckpt_files[0]
    cfg.data.root = '../' + cfg.data.root
    model = Autoencoder.load_from_checkpoint(ckpt_path)
    data = np.load(f"{cfg.data.root}/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)
    with torch.no_grad():
        model.eval()
        x = torch.tensor(data['data'], dtype=torch.float32, device=model.device)
        z = model.encoder(x)
        xh = model.decoder(z)
        zhh = model.encoder(xh)
    zc = z.cpu().numpy()
    # fig = scatter(zc, s=2, alpha=0.2, title='latent', filename=f'{results_path}/latent.html')
    # wandb.log({"latent": wandb.Plotly(fig)})
    # fig = scatter(zhh, s=2, alpha=0.2, title='latent_reembedd', filename=f'{results_path}/latent_reembedded.html')
    # wandb.log({"latent_reembedded": wandb.Plotly(fig)})
    # fig = scatter(x, s=2, alpha=0.2, title='data', filename=f'{results_path}/data.html')
    # wandb.log({"data": wandb.Plotly(fig)})
    # fig = scatter(xh, s=2, alpha=0.2, title='reconstruction', filename=f'{results_path}/reconstruction.html')
    # wandb.log({"reconstruction": wandb.Plotly(fig)})

    # sweep_id = cfg_main.disc_sweep_id
    # sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    # # Initialize an empty list to store run data
    # runs_data = []

    # # Iterate through each run in the sweep
    # for run in sweep.runs:
    #     # Extract metrics and configs
    #     metrics = run.summary._json_dict
    #     configs = run.config
        
    #     # Combine metrics and configs, and add run ID
    #     combined_data = {**metrics, **configs, "run_id": run.id}
        
    #     # Append the combined data to the list
    #     runs_data.append(combined_data)

    # # Create a DataFrame from the runs data
    # df = pd.DataFrame(runs_data)
    # run_ids = df[(df['data.name'] == data_name) & (df['loss.weights.pos1'] == 1.)][['run_id']]
    # assert len(run_ids) == 1
    # run_id = run_ids.iloc[0]
    # run = api.run(f"{entity}/{project}/{run_ids.iloc[0].values[0]}")
    # folder_path = '../../src/wandb/'
    # cfg = OmegaConf.create(run.config)
    # folder_list = glob.glob(f"{folder_path}*{run.id}*")
    # ckpt_files = glob.glob(f"{folder_list[0]}/files/*.ckpt")
    # ckpt_path = ckpt_files[0]
    # cfg.data.root = '../' + cfg.data.root
    # wd = WDiscriminator.load_from_checkpoint(ckpt_path)
    # data = np.load(f"{cfg.data.root}/{cfg.data.name}{cfg.data.filetype}", allow_pickle=True)
    # with torch.no_grad():
    #     wd.eval()
    #     x_disc = torch.tensor(data['data'], dtype=torch.float32, device=wd.device)
    #     probab = wd(x_disc).flatten()
    #     z = model.encoder(x_disc)

    # # title = f'{data_name} {noise_type} {noise_level} {dist_mask}'
    # # scatter(zc, data['mask_x'], s=2, alpha=0.2, title=title, filename=filename).show()
    # # scatter(zhh, data['mask_x'], s=2, alpha=0.2, title=title, filename=filename).show()
    # # scatter(x_disc, data['mask_x'], s=2, alpha=0.2, title=f'{data_name}: input').show()
    # # scatter(x_disc, probab, s=2, alpha=0.2, title=f'{data_name}: probability').show()#, filename='hsphere_disc_prob.html').show()
    # # scatter(x_disc, probab > 0.5, s=2, alpha=0.2, title=f'{data_name}: classification').show()

    # # if plot_recon:
    # #     title = f'[RECON] {title}'
    # #     if savefolder is not None:
    # #         filename = f"{savefolder}/{title1}.html"
    # #     # scatter(xh, data['mask_x'], s=2, alpha=0.2, title=title1, filename=filename).show()

    # max_prob = probab.max()


    xbatch = torch.tensor(data['start_points'], dtype=x.dtype, device=x.device)
    xendbatch = torch.tensor(data['end_points'], dtype=x.dtype, device=x.device)

    # xbatch = model.encoder.preprocessor.normalize(xbatch)
    # xendbatch = model.encoder.preprocessor.normalize(xendbatch)
    ids = torch.zeros((xbatch.size(0),1))
    # ids = torch.eye((xbatch.size(0)))

    dataset = TensorDataset(xbatch, xendbatch, ids)
    dataloader = DataLoader(dataset, batch_size=len(z), shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # wd = wd.to(device)
    # med = probab.quantile(1-data['mask_x'].mean()).to(device)
    # discriminator = discriminator.to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    # for param in wd.parameters():
    #     param.requires_grad = False
    enc_func = lambda x: model.encoder(x)
    # disc_func = lambda x: wd(x).flatten()
    # disc_func_pen = lambda x: 1 - (max_prob - disc_func(x))
    # discriminator_func_for_grad = lambda x: wd(x)
    # discriminator_func_for_grad = lambda x: discriminator.positive_proba(x).reshape(-1,1)
    # disc_func = lambda x: (wd(x).flatten()-probab.min())/(probab.max()-probab.min())
    # disc_func = lambda x: (torch.clamp(wd(x).flatten(), max=med) - probab.min())/(med-probab.min())
    # disc_func_pen = disc_func
    # discriminator_func_for_grad = lambda x: (wd(x)-probab.min())/(probab.max()-probab.min())
    # ofm = offmanifolder_maker(enc_func, disc_func, disc_factor=0.5, max_prob=max_prob)
    gbmodel = GeodesicBridgeOverfit(
        # func=ofm,
        func = enc_func,
        # discriminator_func=disc_func_pen,
        # discriminator_func_for_grad=discriminator_func_for_grad,
        input_dim=x.size(1), 
        hidden_dim=64, 
        scale_factor=1, 
        symmetric=True, 
        num_layers=3, 
        n_tsteps=100, 
        lr=1e-3, 
        weight_decay=1e-3,
        discriminator_weight=0.,
        discriminator_func_for_grad_weight=0.,
        id_dim=1,
        id_emb_dim=1,
        density_weight=0.,
        length_weight=1.,
        data_pts=x,
    )

    if cfg_main.use_wandb:
        logger = WandbLogger()
        checkpoint_dir = wandb.run.dir  # Use wandb's run directory for saving checkpoints

    else:
        logger = TensorBoardLogger(save_dir=os.path.join(results_path))
        checkpoint_dir = results_path  # Use a local directory for saving checkpoints


    gbmodel.lr=1e-3
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg_main.max_epochs,
        log_every_n_steps=20,
        accelerator='cuda',
    )
    trainer.fit(gbmodel, dataloader)
    trainer.save_checkpoint(f"{checkpoint_dir}/gbmodel.ckpt")

    x0, x1, ids = next(iter(dataloader))
    x0 = x0.to(device)
    x1 = x1.to(device)
    ids = ids.to(device)
    x = x.to(device)
    gbmodel = gbmodel.to(device)
    with torch.no_grad():
        xhat = gbmodel(x0, x1, gbmodel.ts.to(device), ids)

    xshape = xhat.shape
    xflatten = xhat.flatten(0,1)
    with torch.no_grad():
        zhat_flatten = model.encoder(xflatten)
        zhat = zhat_flatten.reshape(xshape[0], xshape[1], -1)
        z0 = model.encoder(x0)
        z1 = model.encoder(x1)

        fig = scatter(x.cpu().numpy(), s=2)
        fig = scatter(x0.detach().cpu().numpy(), s=10, fig=fig)
        fig = scatter(x1.detach().cpu().numpy(), s=10, fig=fig)
        fig = trajectories(xhat.detach().cpu().numpy(), s=5, fig=fig, title='Geodesic', filename='geodesic_ofm.html')
    if cfg_main.use_wandb:
        wandb.log({"geodesic": wandb.Plotly(fig)})
    
    fig = scatter(z.cpu().numpy(), s=2)
    fig = scatter(z0.detach().cpu().numpy(), s=10, fig=fig)
    fig = scatter(z1.detach().cpu().numpy(), s=10, fig=fig)
    fig = trajectories(zhat.detach().cpu().numpy(), s=5, fig=fig, title='Geodesic latent', filename=f'{results_path}/geodesic_latent.html')
    if cfg_main.use_wandb:
        wandb.log({"geodesic_latent": wandb.Plotly(fig)})

    with torch.no_grad():
        xconc = xhat.flatten(0, 1)
        probs = disc_func(xconc)
    fig = scatter(xconc.cpu().numpy(), probs.cpu().numpy(), s=2, title='geodesic on-mfd probability', filename=f'{results_path}/geodesic_on_mfd_prob.html')
    if cfg_main.use_wandb:
        wandb.log({"geodesic_on_mfd_prob": wandb.Plotly(fig)})

    if cfg_main.use_wandb:
        run_main.finish()

if __name__ == '__main__':
    main()