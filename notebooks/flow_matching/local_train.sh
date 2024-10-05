#!/bin/bash

conda activate yale529

python local_flow.py \
    --max_epochs 50 \
    --visualize_training \
    --neg_method add \
    --num_samples 64 \
    --batch_size 32 \
    --noise_levels 0.5 1.0 \
    --sampling_rejection \
    --sampling_rejection_method sugar \
    --sampling_rejection_threshold .2 \
    --disc_batch_size 128 \
    --disc_layer_widths 256 128 64 \
    --disc_factor 1000 \
    --disc_max_epochs 100 \
    --fixed_pot \
    --show_plot \
    # --disc_use_gp \
