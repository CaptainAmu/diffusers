"""
FKC LoRA composition: generate K images in batches of B, save to output/images/
and append metadata to output/results.csv (append-only, no overwrite).
"""

import os
import csv
import gc
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import is_peft_available

from config import (
    SCRIPT_DIR,
    OUTPUT_BASE,
    IMAGES_DIR,
    RESULTS_CSV,
    LOG_W_PROGRESS_CSV,
    ADAPTER_A,
    ADAPTER_B,
    STRENGTH_A,
    STRENGTH_B,
    PROMPT_A,
    PROMPT_B,
    PROMPT_C,
    NEGATIVE_PROMPT,
    K,
    B,
    HEIGHT,
    WIDTH,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    BETA,
)
from fkc_sampling import (
    initialize_particles_and_weights,
    step_one_fkc,
    decode_latents_to_pil_list,
)


CSV_COLUMNS = [
    "image_name",
    "prompt_a",
    "prompt_b",
    "prompt_c",
    "negative_prompt",
    "guidance_scale",
    "beta",
    "log_weight",
    "Adapter_A",
    "Adapter_B",
    "StrengthA",
    "StrengthB",
]

# log_w_progress.csv: image_name, Step_1, ..., Step_{NUM_INFERENCE_STEPS}
LOG_W_PROGRESS_COLUMNS = ["image_name"] + [f"Step_{j}" for j in range(1, NUM_INFERENCE_STEPS + 1)]


def _next_image_index(images_dir):
    """Return the next image index (1-based). If 1.png..20.png exist, return 21."""
    os.makedirs(images_dir, exist_ok=True)
    indices = []
    for f in os.listdir(images_dir):
        try:
            # support "1.png" or "1"
            base = os.path.splitext(f)[0]
            if base.isdigit():
                indices.append(int(base))
        except Exception:
            continue
    return max(indices, default=0) + 1


def _append_csv_rows(csv_path, rows):
    """Append rows to CSV; write header if file is new."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def _append_log_w_progress_rows(csv_path, rows, fieldnames):
    """Append rows to log_w_progress.csv; write header if file is new."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def main():
    # --- Output setup (append mode) ---
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    next_id = _next_image_index(IMAGES_DIR)
    print(f"Output: {IMAGES_DIR}, CSV: {RESULTS_CSV}, log_w_progress: {LOG_W_PROGRESS_CSV}")
    print(f"Next image index: {next_id} (will generate {K} images in batches of {B})")

    # --- HF token ---
    token_file = os.path.join(SCRIPT_DIR, "HF_token.txt")
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            os.environ["HF_TOKEN"] = f.read().strip()
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN unset. Put token in HF_token.txt or set HF_TOKEN.")
    print("HF_TOKEN loaded.")

    if not is_peft_available():
        raise RuntimeError("PEFT required: pip install peft")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else "CPU"
    print(f"Device: {gpu_name}")

    # --- Load pipeline and LoRAs ---
    print("\nLoading SDXL pipeline and LoRAs...")
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    if "H100" in gpu_name:
        pipeline.to("cuda")
    elif "A10" in gpu_name or "3090" in gpu_name or "4090" in gpu_name:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.enable_sequential_cpu_offload()

    adapter_name_a, adapter_repo_a = ADAPTER_A
    adapter_name_b, adapter_repo_b = ADAPTER_B
    pipeline.load_lora_weights(adapter_repo_a, adapter_name=adapter_name_a)
    pipeline.load_lora_weights(adapter_repo_b, adapter_name=adapter_name_b)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # --- Encode prompts (A, B, C with respective adapters) ---
    print("Encoding prompts A, B, C...")
    pipeline.set_adapters([adapter_name_a], adapter_weights=[1.0])
    (
        prompt_embeds_a_pos,
        negative_prompt_embeds_a,
        pooled_prompt_embeds_a,
        negative_pooled_prompt_embeds_a,
    ) = pipeline.encode_prompt(
        prompt=PROMPT_A,
        negative_prompt=NEGATIVE_PROMPT,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        device=device,
    )
    pipeline.set_adapters([adapter_name_b], adapter_weights=[1.0])
    (
        prompt_embeds_b_pos,
        negative_prompt_embeds_b,
        pooled_prompt_embeds_b,
        negative_pooled_prompt_embeds_b,
    ) = pipeline.encode_prompt(
        prompt=PROMPT_B,
        negative_prompt=NEGATIVE_PROMPT,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        device=device,
    )

    prompt_embeds_a = torch.cat([negative_prompt_embeds_a, prompt_embeds_a_pos], dim=0)
    prompt_embeds_b = torch.cat([negative_prompt_embeds_b, prompt_embeds_b_pos], dim=0)
    add_text_embeds_a = torch.cat([negative_pooled_prompt_embeds_a, pooled_prompt_embeds_a], dim=0)
    add_text_embeds_b = torch.cat([negative_pooled_prompt_embeds_b, pooled_prompt_embeds_b], dim=0)

    add_time_ids = pipeline._get_add_time_ids(
        original_size=(HEIGHT, WIDTH),
        crops_coords_top_left=(0, 0),
        target_size=(HEIGHT, WIDTH),
        dtype=prompt_embeds_a.dtype,
        text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
    )
    add_time_ids = add_time_ids.to(device)
    add_time_ids_cfg = add_time_ids.repeat(2, 1)

    # --- Scheduler timesteps ---
    pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipeline.scheduler.timesteps
    sigmas = pipeline.scheduler.sigmas.to(device)
    pipeline.scheduler._step_index = None

    # --- Generate K images in batches of B ---
    remaining = K
    current_id = next_id
    total_generated = 0

    while remaining > 0:
        batch_size = min(B, remaining)
        print(f"\n--- Batch: generating {batch_size} images (ids {current_id}..{current_id + batch_size - 1}) ---")

        pipeline.scheduler._step_index = None
        generator = torch.Generator(device=device).manual_seed(42 + total_generated)

        latents, log_weights = initialize_particles_and_weights(
            pipeline,
            device,
            generator,
            prompt_embeds_a.dtype,
            batch_size,
            HEIGHT,
            WIDTH,
        )

        log_weights_history = []
        for step_index in range(NUM_INFERENCE_STEPS):
            latents, log_weights = step_one_fkc(
                latents,
                log_weights,
                step_index,
                pipeline,
                device,
                timesteps,
                sigmas,
                prompt_embeds_a,
                prompt_embeds_b,
                add_text_embeds_a,
                add_text_embeds_b,
                add_time_ids_cfg,
                adapter_name_a,
                adapter_name_b,
                STRENGTH_A,
                STRENGTH_B,
                BETA,
                GUIDANCE_SCALE,
                verbose=True,
            )
            log_weights_history.append(log_weights.cpu().clone().numpy())

        pil_list = decode_latents_to_pil_list(latents, pipeline, vae_batch_size=4)
        log_weights_np = log_weights.cpu().numpy()

        for i in range(batch_size):
            name = f"{current_id + i}"
            path = os.path.join(IMAGES_DIR, f"{name}.png")
            pil_list[i].save(path)
            row = {
                "image_name": f"{name}.png",
                "prompt_a": PROMPT_A,
                "prompt_b": PROMPT_B,
                "prompt_c": PROMPT_C,
                "negative_prompt": NEGATIVE_PROMPT,
                "guidance_scale": GUIDANCE_SCALE,
                "beta": BETA,
                "log_weight": float(log_weights_np[i]),
                "Adapter_A": adapter_repo_a,
                "Adapter_B": adapter_repo_b,
                "StrengthA": STRENGTH_A,
                "StrengthB": STRENGTH_B,
            }
            _append_csv_rows(RESULTS_CSV, [row])

            log_w_row = {"image_name": f"{name}.png"}
            for j in range(NUM_INFERENCE_STEPS):
                log_w_row[f"Step_{j + 1}"] = float(log_weights_history[j][i])
            _append_log_w_progress_rows(LOG_W_PROGRESS_CSV, [log_w_row], LOG_W_PROGRESS_COLUMNS)

        current_id += batch_size
        total_generated += batch_size
        remaining -= batch_size

        del latents, log_weights, pil_list, log_weights_history
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Generated {K} images: {next_id}.png .. {current_id - 1}.png")
    print(f"CSV: {RESULTS_CSV} (appended {K} rows)")
    print(f"Log-weight progress: {LOG_W_PROGRESS_CSV} (appended {K} rows, Step_1..Step_{NUM_INFERENCE_STEPS})")


if __name__ == "__main__":
    main()
