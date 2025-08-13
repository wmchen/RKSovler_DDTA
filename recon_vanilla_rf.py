import numpy as np
import matplotlib.pyplot as plt
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src import FlowMatchEulerDiscreteInversionScheduler, VanillaFluxPipeline
from src.utils import pil2tensor


pretrained_model_name_or_path = "/home/ailab/model_weights_nas/flux/FLUX.1-dev/"
# pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"

scheduler = FlowMatchEulerDiscreteInversionScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
pipe = VanillaFluxPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, scheduler=scheduler)
pipe.to("cuda")

ori_image = Image.open("asset/alley.jpg").convert("RGB").resize((1024, 1024))
prompt = "A narrow alley with building in the background."
inv_result = pipe.inverse(
    image=ori_image,
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=1.0,
)
recon_image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=1.0,
    latents=inv_result.init_noise
).images[0]

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(1, 2)
axs[0].imshow(ori_image)
axs[0].set_title("Origin")
axs[1].imshow(recon_image)
axs[1].set_title("Reconstruction")
plt.savefig("recon_vanilla.jpg", bbox_inches="tight")

psnr_score = psnr(np.array(ori_image), np.array(recon_image))
ssim_score = ssim(np.array(ori_image), np.array(recon_image), win_size=7, channel_axis=2)
lpips_loss = lpips.LPIPS(net="alex")
lpips_score = lpips_loss(pil2tensor(ori_image), pil2tensor(recon_image)).item()
print(f"PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}")
