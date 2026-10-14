import argparse
import math
import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.dataset import LatentLMDBDataset
from utils.distributed import launch_distributed_job
from utils.ode_generation import (
    CausalODETrajectoryGenerator,
    merge_cfg_prompt_embeds,
)
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
DEFAULT_NUM_INFERENCE_STEPS = 48
DEFAULT_SCHEDULER_SHIFT = 5.0
DEFAULT_TARGET_NUM_FRAMES = 21
DEFAULT_TRAJECTORY_INDICES = [0, 6, 12, 18, -2, -1]
DEFAULT_TRAJECTORY_INDICES2 = [24, 30, 36, 42, -2, -1]


def normalize_generator_state_dict(state_dict: dict) -> dict:
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]

    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        fixed[k] = v
    return fixed


def init_model(
    device,
    num_frame_per_block: int,
    scheduler_shift: float,
    num_inference_steps: int,
    negative_prompt: str,
):
    model = WanDiffusionWrapper(is_causal=True).to(device).to(torch.float32)
    model.model.num_frame_per_block = num_frame_per_block
    encoder = WanTextEncoder().to(device).to(torch.float32)

    scheduler = FlowMatchScheduler(
        shift=scheduler_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        denoising_strength=1.0,
    )
    scheduler.sigmas = scheduler.sigmas.to(device)

    unconditional_dict = encoder(text_prompts=[negative_prompt])
    return model, encoder, scheduler, unconditional_dict


def prepare_clean_latent(
    sample: dict,
    target_num_frames: int | None,
    device,
) -> torch.Tensor:
    clean_latent = sample["clean_latent"].to(device).unsqueeze(0)
    if target_num_frames is None:
        return clean_latent

    if clean_latent.shape[1] < target_num_frames:
        raise ValueError(
            "clean_latent has fewer frames than requested: "
            f"{clean_latent.shape[1]} < {target_num_frames}"
        )
    if clean_latent.shape[1] != target_num_frames:
        clean_latent = clean_latent[:, :target_num_frames, ...]
    return clean_latent.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--rawdata_path", type=str, required=True)
    parser.add_argument("--generator_ckpt", type=str, required=True)
    parser.add_argument("--num_frames_per_chunk", default=3)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="blockwise_kv",
        choices=["full", "blockwise_kv"],
    )

    args = parser.parse_args()

    launch_distributed_job()
    global_rank = dist.get_rank()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(
        device=device,
        num_frame_per_block=args.num_frames_per_chunk,
        scheduler_shift=DEFAULT_SCHEDULER_SHIFT,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )
    state_dict = torch.load(args.generator_ckpt, map_location="cpu")
    model.model.load_state_dict(
        normalize_generator_state_dict(state_dict),
        strict=True,
    )

    dataset = LatentLMDBDataset(
        args.rawdata_path,
        max_pair=int(1e8),
    )

    output_folder_1 = os.path.join(args.output_folder, "slow")
    output_folder_2 = os.path.join(args.output_folder, "fast")

    if global_rank == 0:
        os.makedirs(output_folder_1, exist_ok=True)
        os.makedirs(output_folder_2, exist_ok=True)

    trajectory_generator = CausalODETrajectoryGenerator(
        model=model,
        scheduler=scheduler,
        num_frame_per_block=args.num_frames_per_chunk,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale=args.guidance_scale,
    )

    total_steps = int(math.ceil(len(dataset) / dist.get_world_size()))
    for index in tqdm(
        range(total_steps), disable=(global_rank != 0),
    ):
        prompt_index = index * dist.get_world_size() + global_rank
        if prompt_index >= len(dataset):
            continue

        output_path_1 = os.path.join(output_folder_1, f"{prompt_index:05d}.pt")
        output_path_2 = os.path.join(output_folder_2, f"{prompt_index:05d}.pt")

        sample = dataset[prompt_index]
        prompt = sample["prompts"]
        clean_latent = prepare_clean_latent(
            sample=sample,
            target_num_frames=DEFAULT_TARGET_NUM_FRAMES,
            device=device,
        )

        conditional_dict = encoder(text_prompts=[prompt])
        paired_conditional_dict = merge_cfg_prompt_embeds(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )
        initial_noise = torch.randn_like(clean_latent, dtype=torch.float32)
        
        stored_data_list = trajectory_generator.generate(
            clean_latent=clean_latent,
            paired_conditional_dict=paired_conditional_dict,
            trajectory_indices_list=[DEFAULT_TRAJECTORY_INDICES, DEFAULT_TRAJECTORY_INDICES2],
            generation_mode=args.generation_mode,
            initial_noise=initial_noise,
        )
        
        stored_data_1, stored_data_2 = stored_data_list

        torch.save(
            {prompt: stored_data_1.cpu().detach()},
            output_path_1,
        )
        torch.save(
            {prompt: stored_data_2.cpu().detach()},
            output_path_2,
        )

    dist.barrier()


if __name__ == "__main__":
    main()