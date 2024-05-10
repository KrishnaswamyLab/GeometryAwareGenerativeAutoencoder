cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/src; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/geosink/bin/python train.py \
  encoder.layer_widths='[16,16]' \
  encoder.batch_norm=true \
  encoder.dropout=0.2 \
  decoder.layer_widths='[16,16]' \
  decoder.batch_norm=true \
  decoder.dropout=0.2 \
  training.mode='discriminator' \
  training.max_epochs=3 \
  loss.dist_mse_decay=0.2 \
  loss.weights.dist=1 \
  loss.weights.reconstr=1 \
  loss.weights.cycle=1 \
  loss.weights.cycle_dist=1 \
  loss.weights.negative=1 \
  data.root='../toy_data/converted' \
  data.name='make_mix_surface_masked' \
  logger.use_wandb=false
