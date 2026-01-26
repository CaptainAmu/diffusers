import os
import diffusers
import torch
from diffusers.utils import is_peft_available
from diffusers import DiffusionPipeline


token_file = os.path.join(os.path.dirname(__file__), "HF_token.txt")
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
if not os.environ.get("HF_TOKEN"):
    raise ValueError("HF_TOKEN unset! Please ensure you have a local file 'HF_token.txt' which contains the token.")
print("✅ HF_TOKEN successfully loaded from local file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)



print(f"Is PEFT available? {is_peft_available()}")
# This MUST return True for load_lora_weights(adapter_name=...) to work

# 1. 初始化 Pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
gpu_name = torch.cuda.get_device_name(0).upper()
print(f"检测到 GPU: {gpu_name}")
if "H100" in gpu_name:
    print("检测到 H100: 开启全速模式 (pipeline.to('cuda'))")
    pipeline.to("cuda")
elif "A10" in gpu_name or "3090" in gpu_name or "4090" in gpu_name:
    print("检测到 24GB 级显卡: 开启平衡模式 (enable_model_cpu_offload)")
    pipeline.enable_model_cpu_offload()
else:
    print("检测到显存较小或未知显卡: 开启保守模式 (enable_sequential_cpu_offload)")
    # 最省显存模式，甚至可以在 8GB 显卡跑 SDXL
    pipeline.enable_sequential_cpu_offload()


# 2. 一次性加载所有 LoRA (给它们起个名字)
pipeline.load_lora_weights('Fictiverse/Voxel_XL_Lora', adapter_name="voxel")
pipeline.load_lora_weights('ntc-ai/SDXL-LoRA-slider.sexy', adapter_name="sexy")

# 准备输出目录
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# --- 实验 1: 只用 Voxel ---
pipeline.set_adapters(["voxel"], adapter_weights=[1.0])
img1 = pipeline("A japanese animation girl eating a hamburger, voxel style").images[0]
img1.save(os.path.join(output_dir, "only_voxel.png"))

# --- 实验 2: 只用 Sexy ---
pipeline.set_adapters(["sexy"], adapter_weights=[1.0])
img2 = pipeline("A japanese animation girl eating a hamburger, sexy style").images[0]
img2.save(os.path.join(output_dir, "only_sexy.png"))

# --- 实验 3: 两个 LoRA 混合 (各占一半) ---
pipeline.set_adapters(["voxel", "sexy"], adapter_weights=[0.5, 0.5])
img3 = pipeline("A japanese animation girl eating a hamburger, voxel and sexy style").images[0]
img3.save(os.path.join(output_dir, "mixed_loras.png"))

# --- 实验 4: 禁用所有 LoRA (回到原图) ---
pipeline.disable_lora()
img4 = pipeline("A japanese animation girl eating a hamburger").images[0]
img4.save(os.path.join(output_dir, "no_lora.png"))