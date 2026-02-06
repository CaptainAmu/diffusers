"""
Configuration for FKC (Feynman-Kac Corrected) LoRA composition using SDXL 1.0 base model.
Edit K, B, beta here or pass via CLI in main.py.
"""

import os

# Script directory (output will be under this folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "output")
IMAGES_DIR = os.path.join(OUTPUT_BASE, "images")
RESULTS_CSV = os.path.join(OUTPUT_BASE, "results.csv")
LOG_W_PROGRESS_CSV = os.path.join(OUTPUT_BASE, "log_w_progress.csv")

# Adapters: (adapter_name, huggingface_repo_id)
ADAPTER_A = ("anime", "ntc-ai/SDXL-LoRA-slider.anime")
ADAPTER_B = ("3D_render", "goofyai/3d_render_style_xl")
STRENGTH_A = 1
STRENGTH_B = 1

# Prompts
PROMPT_A = "Round face, buzz cut, chubby skin, full beard, dark skin, East Asian, round-framed glasses, 1.9 meters tall, white socks, gay man."
PROMPT_B = "Round face, buzz cut, chubby skin, full beard, dark skin, East Asian, round-framed glasses, 1.9 meters tall, white socks, gay man."
PROMPT_C = "Round face, buzz cut, chubby skin, full beard, dark skin, East Asian, round-framed glasses, 1.9 meters tall, white socks, gay man."
NEGATIVE_PROMPT = "Slim, skinny, frail, pale skin, fair skin, long hair, feminine features, shredded, ripped (too lean), six-pack abs (too defined), low height, short stature, messy beard, eyeglasses, suit, formal wear, blurry, low quality, deformed anatomy, extra fingers"

# Generation: K total images, B per batch during generation (to avoid OOM)
K = 2
B = 2

# Resolution and steps
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50

# Guidance scale: Under each LoRA i, predict score:  adapter_i_pred = guidance * (cond - uncond) + uncond 
# Score mix: (1 - beta) * adapter_a_pred + beta * adapter_b_pred
GUIDANCE_SCALE = 5
BETA = 0.5
