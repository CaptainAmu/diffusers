"""
FKC (Feynman-Kac Correction) sampling: one-step update, init, and VAE decode.
Used by main.py to generate images with (1-beta)*voxel + beta*sexy score mixing.
"""

import torch


def initialize_particles_and_weights(
    pipeline,
    device,
    generator,
    prompt_embeds_dtype,
    K,
    height,
    width,
):
    """Initialize K particles (latents) and zero log-weights."""
    latents = pipeline.prepare_latents(
        batch_size=K,
        num_channels_latents=4,
        height=height,
        width=width,
        dtype=prompt_embeds_dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    log_weights = torch.zeros(K, device=device, dtype=torch.float32)
    return latents, log_weights


def step_one_fkc(
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
    strength_a,
    strength_b,
    beta,
    guidance_scale,
    verbose=True,
):
    """
    One FKC backward step for all K particles.
    Mix: (1-beta)*pred_a + beta*pred_b; update log_weights in-place.
    """
    t = timesteps[step_index]
    sigma_idx = pipeline.scheduler.index_for_timestep(t)
    sigma_t = sigmas[sigma_idx].item() if isinstance(sigmas[sigma_idx], torch.Tensor) else sigmas[sigma_idx]
    if sigma_idx < len(sigmas) - 1:
        sigma_next = (
            sigmas[sigma_idx + 1].item()
            if isinstance(sigmas[sigma_idx + 1], torch.Tensor)
            else sigmas[sigma_idx + 1]
        )
        dt = abs(sigma_t - sigma_next)
    else:
        dt = abs(sigma_t)

    mixed_list = []
    K = latents.shape[0]
    unet_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for k in range(K):
            latent_k = latents[k : k + 1]
            latent_cfg = latent_k.repeat(2, 1, 1, 1)
            latent_cfg = pipeline.scheduler.scale_model_input(latent_cfg, t)
            latent_cfg = latent_cfg.to(unet_device)

            added_cond_a = {"text_embeds": add_text_embeds_a, "time_ids": add_time_ids_cfg}
            pipeline.set_adapters([adapter_name_a], adapter_weights=[strength_a])
            out_a = pipeline.unet(
                latent_cfg,
                t,
                encoder_hidden_states=prompt_embeds_a,
                added_cond_kwargs=added_cond_a,
                return_dict=False,
            )[0]
            pred_a_uncond, pred_a_text = out_a.chunk(2)
            pred_a = pred_a_uncond + guidance_scale * (pred_a_text - pred_a_uncond)
            del out_a, pred_a_uncond, pred_a_text

            added_cond_b = {"text_embeds": add_text_embeds_b, "time_ids": add_time_ids_cfg}
            pipeline.set_adapters([adapter_name_b], adapter_weights=[strength_b])
            out_b = pipeline.unet(
                latent_cfg,
                t,
                encoder_hidden_states=prompt_embeds_b,
                added_cond_kwargs=added_cond_b,
                return_dict=False,
            )[0]
            pred_b_uncond, pred_b_text = out_b.chunk(2)
            pred_b = pred_b_uncond + guidance_scale * (pred_b_text - pred_b_uncond)
            del out_b, pred_b_uncond, pred_b_text, latent_cfg

            mixed_k = (1 - beta) * pred_a + beta * pred_b
            mixed_list.append(mixed_k)

            dist_sq_k = torch.sum((pred_a - pred_b) ** 2, dim=[1, 2, 3]).squeeze(0)
            weight_update_k = (0.5 * beta * (beta - 1) * dist_sq_k * dt).to(log_weights.dtype)
            log_weights[k] = log_weights[k] + weight_update_k.item()

            del pred_a, pred_b, mixed_k, dist_sq_k, weight_update_k

        mixed = torch.cat(mixed_list, dim=0)
        del mixed_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pipeline.scheduler._step_index = sigma_idx
        latents_next = pipeline.scheduler.step(mixed, t, latents, return_dict=False)[0]
        del mixed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if verbose:
        print(
            f"Step {step_index + 1}/{len(timesteps)} t={t.item():.0f} σ_t={sigma_t:.4f} dt={dt:.4f} | "
            f"log_weights {log_weights.cpu().numpy()} | Latents shape: {latents_next.shape}"
        )

    return latents_next, log_weights


def decode_latents_to_pil_list(latents, pipeline, vae_batch_size=4):
    """
    Decode [K, 4, H, W] latents to K PIL images.
    Uses FP32 and vae_batch_size chunks to avoid OOM.
    """
    original_vae_dtype = pipeline.vae.dtype
    pipeline.vae.to(dtype=torch.float32)

    vae_device = "cuda" if torch.cuda.is_available() else "cpu"
    latents = latents.to(device=vae_device, dtype=torch.float32)
    K = latents.shape[0]

    with torch.no_grad():
        scaling_factor = pipeline.vae.config.scaling_factor
        shift_factor = getattr(pipeline.vae.config, "shift_factor", None)
        if shift_factor is not None:
            latents_to_decode = (latents / scaling_factor) + shift_factor
        else:
            latents_to_decode = latents / scaling_factor

        decoded_list = []
        for start in range(0, K, vae_batch_size):
            end = min(start + vae_batch_size, K)
            chunk = latents_to_decode[start:end]
            out = pipeline.vae.decode(chunk, return_dict=False)[0]
            decoded_list.append(out)
        decoded = torch.cat(decoded_list, dim=0)

        pil_images = pipeline.image_processor.postprocess(decoded, output_type="pil")

    pipeline.vae.to(dtype=original_vae_dtype)
    return pil_images
