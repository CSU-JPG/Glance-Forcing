Builing on [Rolling Forcing](https://github.com/TencentARC/RollingForcing), we implemented minute-level long video generation.

### Installation

```bash
conda activate causal_forcing
pip install tensorboard opencv-python packaging
```

### CLI inference
```
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download zhuhz22/Causal-Forcing chunkwise/longvideo.pt --local-dir ../checkpoints

python inference.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --output_folder videos/rolling_forcing_dmd \
    --checkpoint_path ../checkpoints/chunkwise/longvideo.pt \
    --data_path prompts/example_prompts.txt \
    --num_output_frames 252 \
    --use_ema
```

### Training
```
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B

torchrun --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  train.py \
  -- \
  --config_path configs/rolling_forcing_dmd.yaml \
  --logdir logs/rolling_forcing_dmd
```
> We recommend training for 3000 steps.

### Acknowledge
We adopt [Rolling Forcing](https://github.com/TencentARC/RollingForcing) as our long video generation framework and only change the ODE initialization part.
