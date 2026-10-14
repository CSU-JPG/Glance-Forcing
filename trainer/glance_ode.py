import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle
from model.ode import ODERegression
from collections import defaultdict
from utils.misc import (
    set_seed
)
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm, trange

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = True
        self.world_size = 1
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            config.seed = random_seed.item()

        set_seed(config.seed)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.wandb_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer

        assert config.trainer == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)

        self.model.generator = self.model.generator.to(device=self.device, dtype=self.dtype)
        self.model.text_encoder = self.model.text_encoder.to(device=self.device, dtype=self.dtype)

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Initialize the dataloader
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=8)

        self.dataloader = cycle(dataloader)

        sample = dataset[self.config.fixed_sample]

        self.fixed_batch = {
            k: [v] if isinstance(v, str) else v.unsqueeze(0)
            for k, v in sample.items()
        }
        self.step = 0

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
                
            self.model.generator.load_state_dict(state_dict, strict=True)

        lora_config = LoraConfig(
            r = 32,
            lora_alpha = 128,
            init_lora_weights = "gaussian",
            target_modules = "all-linear",
        )

        self.model.generator = get_peft_model(
            self.model.generator,
            lora_config
        )

        self.model.generator.print_trainable_parameters()

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Saving LoRA model states...")
        
        full_state_dict = self.model.generator.state_dict()
        lora_state_dict = get_peft_model_state_dict(
            self.model.generator,
            full_state_dict
        )

        ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "model.pt")
        
        torch.save(lora_state_dict, save_path)
        print("Model saved to", save_path)

    def train(self):
        lora_type = getattr(config, "training_lora")
        if lora_type == "fast":
            train_sigmas = torch.tensor([0.8333, 0.7500, 0.6250, 0.4167], dtype=torch.float32) # fast sigma
        else:
            train_sigmas = torch.tensor([1.0000, 0.9722, 0.9375, 0.8929], dtype=torch.float32) # slow sigma
        print("start training!")
        loss_history = [[] for _ in range(len(train_sigmas))]
        for step in trange(15 * len(train_sigmas)):
            t_idx = step % len(train_sigmas)
            index = torch.full((1, 21), t_idx, dtype=torch.long)
            sigmas = train_sigmas[t_idx]

            # Step 1: Get the fixed batch of text prompts
            batch = self.fixed_batch
            text_prompts = batch["prompts"]
            ode_latent = batch["ode_latent"].to(
                device=self.device, dtype=self.dtype)

            # Step 2: Extract the conditional infos
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

            # Step 3: Train the generator
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision, dtype=torch.bfloat16):
                generator_loss, log_dict = self.model.generator_loss(
                    t_idx=index,
                    sigmas=sigmas,
                    ode_latent=ode_latent,
                    conditional_dict=conditional_dict
                )

            loss_history[t_idx].append(generator_loss.detach().item())

            ode_loss_val = log_dict["ode_loss"].detach().cpu().item()

            self.generator_optimizer.zero_grad()
            (generator_loss * 1).backward()
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(), 
                self.max_grad_norm
            )
            self.generator_optimizer.step()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1

            if self.step > 0 and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()
