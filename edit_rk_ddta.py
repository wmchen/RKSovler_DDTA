import matplotlib.pyplot as plt
import torch
from PIL import Image

from src import FlowMatchEulerDiscreteInversionScheduler, RKSolverFluxPipeline


r = 4
pretrained_model_name_or_path = "/home/ailab/model_weights_nas/flux/FLUX.1-dev/"
# pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"

scheduler = FlowMatchEulerDiscreteInversionScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
pipe = RKSolverFluxPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16, scheduler=scheduler)
pipe.to("cuda")

ori_image = Image.open("asset/cake.jpg").convert("RGB").resize((1024, 1024))
edit_image = pipe.edit(
    source_prompt="a [round] shape cake with orange frosting on a wooden plate",
    target_prompt="a [star] shape cake with orange frosting on a wooden plate",
    image=ori_image,
    r=r,
    hook_steps=[0, 1],
    hook_trans_block=True,
    hook_single_trans_block=False,
    hook_ci=True,
    hook_ic=True,
    hook_v=True,
    ci_strategy="replace",
    ic_strategy="replace",
    vc_strategy="keep",
    vi_strategy="mean",
    num_inference_steps=5,
    inversion_guidance_scale=1.0,
    denoise_guidance_scale=3.0,
    in_memory=True
).edit_image

fig = plt.figure(figsize=(20, 10))
axs = fig.subplots(1, 2)
axs[0].imshow(ori_image)
axs[0].set_title("Origin")
axs[1].imshow(edit_image)
axs[1].set_title("Edited")
plt.savefig("edit_rk_ddta.jpg", bbox_inches="tight")
