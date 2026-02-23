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
ADAPTER_B = ("")
STRENGTH_A = 1
STRENGTH_B = 1

# Prompts
PROMPT_A = "DJ Trump having hamburger"
PROMPT_B = ""
PROMPT_C = ""
NEGATIVE_PROMPT = "blurry, low quality, distorted, "

# Generation: K total images, B per batch during generation (to avoid OOM)
K = 2
B = 2

# Resolution and steps
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50

# Guidance scale: Under each LoRA i, predict score:  adapter_i_pred = guidance * (cond - uncond) + uncond 
# Score mix: (1 - beta) * adapter_a_pred + beta * adapter_b_pred
GUIDANCE_SCALE = 6
BETA = 0.9
