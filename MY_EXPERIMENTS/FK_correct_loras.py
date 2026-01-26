"""
Custom SDXL Sampling Loop with FKC (Forward-Kolmogorov Correction)
Implements geometric averaging of two LoRA adapters with importance weighting.
Based on Proposition 3.1 (Classifier-Free Guidance + FKC).

Mathematical Formulation:
- Drift (Mixed Score): dx_t = σ_t^2 * ((1-β) * ∇log q_t^1(x_t) + β * ∇log q_t^2(x_t)) * dt
- Weight Evolution: dw_t = (σ_t^2 / 2) * β(β-1) * ||∇log q_t^1(x_t) - ∇log q_t^2(x_t)||^2 * dt
- Score Relation: ∇log q_t(x_t) ≈ -ε_θ(x_t, t) / σ_t

Implementation Details:
1. Maintains K parallel trajectories (particles)
2. Tracks cumulative log-weights based on score divergence
3. Uses dual-pass inference (one for each LoRA adapter)
4. Applies CFG independently to both predictions
5. Steps with geometric average: (1-β) * noise_a + β * noise_b
6. Selects winning particle based on final importance weights
"""

import os
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import is_peft_available
from PIL import Image


# ============================================================================
# 1. SETUP AND CONFIGURATION
# ============================================================================

# Load HF token if available
token_file = os.path.join(os.path.dirname(__file__), "HF_token.txt")
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
if not os.environ.get("HF_TOKEN"):
    raise ValueError("HF_TOKEN unset! Please ensure you have a local file 'HF_token.txt' which contains the token.")
print("✅ HF_TOKEN successfully loaded from local file.")

# Ensure PEFT is available
if not is_peft_available():
    raise RuntimeError("PEFT is required for LoRA adapter loading. Please install: pip install peft")
print(f"✅ PEFT available: {is_peft_available()}")

# Setup output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Configuration parameters
K = 4  # Number of parallel trajectories (particles)
beta = 0.5  # Mixing coefficient for geometric average
guidance_scale = 7.5  # Classifier-free guidance scale
num_inference_steps = 50  # Number of denoising steps
height = 1024
width = 1024

# Prompts for each style
prompt_a = "A japanese animation girl eating a hamburger, voxel style"
prompt_b = "A japanese animation girl eating a hamburger, sexy style"
negative_prompt = "blurry, low quality, distorted"

# LoRA adapter names (matching Two_loras.py)
adapter_name_a = "voxel"  # style_a
adapter_name_b = "sexy"   # style_b


# ============================================================================
# 2. MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING MODEL")
print("="*80)

# Load SDXL pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)

# GPU detection and device placement
gpu_name = torch.cuda.get_device_name(0).upper()
print(f"Detected GPU: {gpu_name}")

if "H100" in gpu_name:
    print("H100 detected: Using full-speed mode (pipeline.to('cuda'))")
    pipeline.to("cuda")
    device = "cuda"
elif "A10" in gpu_name or "3090" in gpu_name or "4090" in gpu_name:
    print("24GB-class GPU detected: Using balanced mode (enable_model_cpu_offload)")
    pipeline.enable_model_cpu_offload()
    device = "cuda"
else:
    print("Smaller/unknown GPU detected: Using conservative mode (enable_sequential_cpu_offload)")
    pipeline.enable_sequential_cpu_offload()
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LoRA adapters
print(f"\nLoading LoRA adapter '{adapter_name_a}' from 'Fictiverse/Voxel_XL_Lora'...")
pipeline.load_lora_weights('Fictiverse/Voxel_XL_Lora', adapter_name=adapter_name_a)

print(f"Loading LoRA adapter '{adapter_name_b}' from 'ntc-ai/SDXL-LoRA-slider.sexy'...")
pipeline.load_lora_weights('ntc-ai/SDXL-LoRA-slider.sexy', adapter_name=adapter_name_b)

# Set scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
print("✅ Scheduler set to EulerAncestralDiscreteScheduler")


# ============================================================================
# 3. PRE-COMPUTE PROMPT EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("PRE-COMPUTING PROMPT EMBEDDINGS")
print("="*80)

# Encode prompts for style A
print(f"Encoding prompt A: '{prompt_a}'")
pipeline.set_adapters([adapter_name_a], adapter_weights=[1.0])
(
    prompt_embeds_a_pos,
    negative_prompt_embeds_a,
    pooled_prompt_embeds_a,
    negative_pooled_prompt_embeds_a,
) = pipeline.encode_prompt(
    prompt=prompt_a,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    device=device,
)

# Encode prompts for style B
print(f"Encoding prompt B: '{prompt_b}'")
pipeline.set_adapters([adapter_name_b], adapter_weights=[1.0])
(
    prompt_embeds_b_pos,
    negative_prompt_embeds_b,
    pooled_prompt_embeds_b,
    negative_pooled_prompt_embeds_b,
) = pipeline.encode_prompt(
    prompt=prompt_b,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    device=device,
)

# Concatenate negative and positive embeddings for CFG (as done in SDXL pipeline)
# This matches the format expected by the UNet: [negative, positive]
prompt_embeds_a = torch.cat([negative_prompt_embeds_a, prompt_embeds_a_pos], dim=0)
prompt_embeds_b = torch.cat([negative_prompt_embeds_b, prompt_embeds_b_pos], dim=0)
add_text_embeds_a = torch.cat([negative_pooled_prompt_embeds_a, pooled_prompt_embeds_a], dim=0)
add_text_embeds_b = torch.cat([negative_pooled_prompt_embeds_b, pooled_prompt_embeds_b], dim=0)

# Prepare time_ids (required for SDXL)
original_size = (height, width)
crops_coords_top_left = (0, 0)
target_size = (height, width)
add_time_ids = pipeline._get_add_time_ids(
    original_size=original_size,
    crops_coords_top_left=crops_coords_top_left,
    target_size=target_size,
    dtype=prompt_embeds_a.dtype,
    text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
)
add_time_ids = add_time_ids.to(device)

# For CFG, we need to duplicate time_ids
add_time_ids_cfg = add_time_ids.repeat(2, 1)  # [2, 6] for uncond + cond

print(f"✅ Prompt embeddings prepared:")
print(f"   - prompt_embeds_a_pos shape: {prompt_embeds_a_pos.shape}")
print(f"   - negative_prompt_embeds_a shape: {negative_prompt_embeds_a.shape}")
print(f"   - pooled_prompt_embeds_a shape: {pooled_prompt_embeds_a.shape}")
print(f"   - negative_pooled_prompt_embeds_a shape: {negative_pooled_prompt_embeds_a.shape}")
print(f"   - prompt_embeds_a shape (after concat): {prompt_embeds_a.shape} (concatenated: [neg, pos])")
print(f"   - prompt_embeds_b shape (after concat): {prompt_embeds_b.shape} (concatenated: [neg, pos])")
print(f"   - add_text_embeds_a shape (after concat): {add_text_embeds_a.shape} (concatenated: [neg, pos])")
print(f"   - add_text_embeds_b shape (after concat): {add_text_embeds_b.shape} (concatenated: [neg, pos])")


# ============================================================================
# 4. INITIALIZE PARTICLES AND WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING PARTICLES AND WEIGHTS")
print("="*80)

# Initialize latents for K particles
latent_shape = (K, 4, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)
generator = torch.Generator(device=device).manual_seed(42)

# Use pipeline's prepare_latents to get properly scaled initial noise
latents = pipeline.prepare_latents(
    batch_size=K,
    num_channels_latents=4,
    height=height,
    width=width,
    dtype=prompt_embeds_a.dtype,
    device=device,
    generator=generator,
    latents=None,
)

# Initialize log weights (cumulative importance weights)
log_weights = torch.zeros(K, device=device, dtype=torch.float32)

print(f"✅ Initialized {K} particles:")
print(f"   - Latents shape: {latents.shape}")
print(f"   - Log weights shape: {log_weights.shape}")


# ============================================================================
# 5. SETUP SCHEDULER
# ============================================================================

print("\n" + "="*80)
print("SETTING UP SCHEDULER")
print("="*80)

pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipeline.scheduler.timesteps
sigmas = pipeline.scheduler.sigmas.to(device)

# Reset scheduler step index (important for proper state management)
pipeline.scheduler._step_index = None

print(f"✅ Scheduler configured:")
print(f"   - Number of steps: {num_inference_steps}")
print(f"   - Timesteps shape: {timesteps.shape}")
print(f"   - Sigmas shape: {sigmas.shape}")


# ============================================================================
# 6. CUSTOM SAMPLING LOOP
# ============================================================================

print("\n" + "="*80)
print("STARTING CUSTOM SAMPLING LOOP")
print("="*80)
print(f"Parameters: beta={beta}, guidance_scale={guidance_scale}, K={K}")

for i, t in enumerate(timesteps):
    print(f"\nStep {i+1}/{len(timesteps)}: t={t.item()}")
    
    # Get current sigma using scheduler's index_for_timestep
    sigma_idx = pipeline.scheduler.index_for_timestep(t)
    sigma_t = sigmas[sigma_idx].item() if isinstance(sigmas[sigma_idx], torch.Tensor) else sigmas[sigma_idx]
    
    # Compute dt (step size in sigma space)
    # For EulerAncestralDiscreteScheduler, dt = sigma_down - sigma (computed internally in step())
    # We approximate dt as the difference between consecutive sigmas
    # This is a reasonable approximation for the weight update formula
    if sigma_idx < len(sigmas) - 1:
        sigma_next = sigmas[sigma_idx + 1].item() if isinstance(sigmas[sigma_idx + 1], torch.Tensor) else sigmas[sigma_idx + 1]
        dt = abs(sigma_t - sigma_next)  # Step size in sigma space
    else:
        dt = abs(sigma_t)  # Last step
    
    # Scale model input
    latent_model_input = pipeline.scheduler.scale_model_input(latents, t)
    
    # Expand latents for CFG: [K, 4, H, W] -> [K*2, 4, H, W]
    latent_model_input_cfg = latent_model_input.repeat_interleave(2, dim=0)
    
    # Prepare embeddings for CFG with K particles
    # add_text_embeds_a/b are already [2, pooled_dim] (neg, pos) for CFG
    # We need to expand to [K*2, pooled_dim] for K particles
    # Use unsqueeze + expand or repeat_interleave for safer dimension handling
    if add_text_embeds_a.dim() == 1:
        # If 1D, it should be [pooled_dim], but we expect [2, pooled_dim] after concat
        # This shouldn't happen, but handle it just in case
        add_text_embeds_a = add_text_embeds_a.unsqueeze(0)
    if add_text_embeds_b.dim() == 1:
        add_text_embeds_b = add_text_embeds_b.unsqueeze(0)
    
    # Expand: [2, pooled_dim] -> [K*2, pooled_dim]
    # Use repeat_interleave to repeat each row K times: [neg, pos] -> [neg, neg, ..., pos, pos, ...]
    add_text_embeds_cfg_a = add_text_embeds_a.repeat_interleave(K, dim=0)  # [K*2, pooled_dim]
    add_text_embeds_cfg_b = add_text_embeds_b.repeat_interleave(K, dim=0)  # [K*2, pooled_dim]
    
    # prompt_embeds_a/b are already [2, seq_len, hidden] (neg, pos) for CFG
    # We need to expand to [K*2, seq_len, hidden] for K particles
    # Use repeat_interleave to repeat each row K times
    prompt_embeds_cfg_a = prompt_embeds_a.repeat_interleave(K, dim=0)  # [K*2, seq_len, hidden]
    prompt_embeds_cfg_b = prompt_embeds_b.repeat_interleave(K, dim=0)  # [K*2, seq_len, hidden]
    
    # add_time_ids_cfg is [2, 6] (neg, pos) for CFG
    # We need to expand to [K*2, 6] for K particles
    add_time_ids_cfg_batch = add_time_ids_cfg.repeat_interleave(K, dim=0)  # [K*2, 6]
    
    # ========================================================================
    # Dual Pass: Get noise predictions from both LoRA adapters
    # ========================================================================
    
    # Pass A: Use style_a adapter
    pipeline.set_adapters([adapter_name_a], adapter_weights=[1.0])
    added_cond_kwargs_a = {
        "text_embeds": add_text_embeds_cfg_a,
        "time_ids": add_time_ids_cfg_batch,
    }
    noise_pred_a = pipeline.unet(
        latent_model_input_cfg,
        t,
        encoder_hidden_states=prompt_embeds_cfg_a,
        added_cond_kwargs=added_cond_kwargs_a,
        return_dict=False,
    )[0]
    
    # Apply CFG to noise_pred_a
    noise_pred_uncond_a, noise_pred_text_a = noise_pred_a.chunk(2)
    noise_pred_a = noise_pred_uncond_a + guidance_scale * (noise_pred_text_a - noise_pred_uncond_a)
    
    # Pass B: Use style_b adapter
    pipeline.set_adapters([adapter_name_b], adapter_weights=[1.0])
    added_cond_kwargs_b = {
        "text_embeds": add_text_embeds_cfg_b,
        "time_ids": add_time_ids_cfg_batch,
    }
    noise_pred_b = pipeline.unet(
        latent_model_input_cfg,
        t,
        encoder_hidden_states=prompt_embeds_cfg_b,
        added_cond_kwargs=added_cond_kwargs_b,
        return_dict=False,
    )[0]
    
    # Apply CFG to noise_pred_b
    noise_pred_uncond_b, noise_pred_text_b = noise_pred_b.chunk(2)
    noise_pred_b = noise_pred_uncond_b + guidance_scale * (noise_pred_text_b - noise_pred_uncond_b)
    
    # ========================================================================
    # Weight Update: Track importance weights based on score divergence
    # ========================================================================
    
    # Compute squared Euclidean distance between scores
    # Note: ∇log q_t(x_t) ≈ -ε_θ(x_t, t) / σ_t
    # So we compute distance between noise predictions (which are proportional to scores)
    dist_sq = torch.sum((noise_pred_a - noise_pred_b) ** 2, dim=[1, 2, 3])  # [K]
    
    # Update log weights according to Proposition 3.1
    # dw_t = (σ_t^2 / 2) * β(β-1) * ||∇log q_t^1 - ∇log q_t^2||^2 * dt
    # Note: The formula uses σ_t^2, and dt is the step size in the continuous-time SDE
    # beta * (beta - 1) is negative for beta in (0, 1), which is correct for importance weighting
    weight_update = 0.5 * beta * (beta - 1) * dist_sq * (sigma_t ** 2) * dt
    log_weights = log_weights + weight_update.to(log_weights.dtype)
    
    print(f"   - σ_t: {sigma_t:.4f}, dt: {dt:.4f}")
    print(f"   - Mean dist_sq: {dist_sq.mean().item():.4f}")
    print(f"   - Mean weight_update: {weight_update.mean().item():.6f}")
    print(f"   - Log weights range: [{log_weights.min().item():.4f}, {log_weights.max().item():.4f}]")
    
    # ========================================================================
    # Stepping: Use geometric average of scores
    # ========================================================================
    
    # Compute mixed noise: (1-β) * noise_pred_a + β * noise_pred_b
    mixed_noise = (1 - beta) * noise_pred_a + beta * noise_pred_b
    
    # Step the scheduler
    latents = pipeline.scheduler.step(mixed_noise, t, latents, return_dict=False)[0]


# ============================================================================
# 7. SELECT WINNING PARTICLE AND DECODE
# ============================================================================

print("\n" + "="*80)
print("SELECTING WINNING PARTICLE")
print("="*80)

# Convert log weights to probabilities
probabilities = F.softmax(log_weights, dim=0)
print(f"Probabilities: {probabilities.cpu().numpy()}")

# Select winning particle (highest probability)
winning_idx = torch.argmax(probabilities).item()
print(f"✅ Selected particle {winning_idx} with probability {probabilities[winning_idx].item():.4f}")

# Extract winning latent
winning_latent = latents[winning_idx:winning_idx+1]  # Keep batch dimension

# Decode with VAE
print("\nDecoding image with VAE...")
with torch.no_grad():
    image = pipeline.vae.decode(winning_latent / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]

# Save image
output_path = os.path.join(output_dir, "fk_correct_loras_output.png")
image.save(output_path)
print(f"✅ Image saved to: {output_path}")

# Also save all particles for comparison (optional)
print("\nSaving all particles for comparison...")
for k in range(K):
    particle_latent = latents[k:k+1]
    with torch.no_grad():
        particle_image = pipeline.vae.decode(
            particle_latent / pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        particle_image = pipeline.image_processor.postprocess(particle_image, output_type="pil")[0]
    particle_path = os.path.join(output_dir, f"particle_{k}_prob_{probabilities[k].item():.4f}.png")
    particle_image.save(particle_path)

print(f"✅ All particles saved to {output_dir}/")
print("\n" + "="*80)
print("SAMPLING COMPLETE")
print("="*80)
