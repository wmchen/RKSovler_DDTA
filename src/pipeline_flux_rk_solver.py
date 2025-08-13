from dataclasses import dataclass
from typing import Union, Optional, Callable, Dict, List, Any

import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import BaseOutput
from mlcbase import Logger, EmojiProgressBar

from .ddta import DDTA
from .scheduler import FlowMatchEulerDiscreteInversionScheduler
from .utils import EditOutput, ImageLoader, image2latent


BUTCHER = {
    1: {
        "A": torch.Tensor([0.0]),
        "B": torch.Tensor([1.0]),
        "C": torch.Tensor([0.0])
    },
    2: {
        "default": "heun",
        "midpoint": {
            "A": torch.Tensor([[0.0, 0.0], 
                               [1/2, 0.0]]),
            "B": torch.Tensor([0.0, 1.0]),
            "C": torch.Tensor([0.0, 1/2])
        },
        "heun": {
            "A": torch.Tensor([[0.0, 0.0], 
                               [1.0, 0.0]]),
            "B": torch.Tensor([1/2, 1/2]),
            "C": torch.Tensor([0.0, 1.0])
        },
        "ralston": {
            "A": torch.Tensor([[0.0, 0.0], 
                               [2/3, 0.0]]),
            "B": torch.Tensor([1/4, 3/4]),
            "C": torch.Tensor([0.0, 2/3])
        }
    },
    3: {
        "default": "kutta",
        "kutta": {
            "A": torch.Tensor([[0.0, 0.0, 0.0],
                               [1/2, 0.0, 0.0],
                               [-1.0, 2.0, 0.0]]),
            "B": torch.Tensor([1/6, 2/3, 1/6]),
            "C": torch.Tensor([0.0, 1/2, 1])
        },
        "heun": {
            "A": torch.Tensor([[0.0, 0.0, 0.0],
                               [1/3, 0.0, 0.0],
                               [0.0, 2/3, 0.0]]),
            "B": torch.Tensor([1/4, 0.0, 3/4]),
            "C": torch.Tensor([0.0, 1/3, 2/3])
        },
        "ralston": {
            "A": torch.Tensor([[0.0, 0.0, 0.0],
                               [1/2, 0.0, 0.0],
                               [0.0, 3/4, 0.0]]),
            "B": torch.Tensor([2/9, 1/3, 4/9]),
            "C": torch.Tensor([0.0, 1/2, 3/4])
        },
        "VanDerHouwenWray": {
            "A": torch.Tensor([[0.0, 0.0, 0.0],
                               [8/15, 0.0, 0.0],
                               [1/4, 5/12, 0.0]]),
            "B": torch.Tensor([1/4, 0.0, 3/4]),
            "C": torch.Tensor([0.0, 8/15, 2/3])
        },
        "SSPRK3": {
            "A": torch.Tensor([[0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [1/4, 1/4, 0.0]]),
            "B": torch.Tensor([1/6, 1/6, 2/3]),
            "C": torch.Tensor([0.0, 1.0, 1/2])
        },
    },
    4: {
        "default": "3/8-rule",
        "classic": {
            "A": torch.Tensor([[0.0, 0.0, 0.0, 0.0],
                               [1/2, 0.0, 0.0, 0.0],
                               [0.0, 1/2, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]]),
            "B": torch.Tensor([1/6, 1/3, 1/3, 1/6]),
            "C": torch.Tensor([0.0, 1/2., 1/2., 1.0])
        },
        "3/8-rule": {
            "A": torch.Tensor([[0.0, 0.0, 0.0, 0.0],
                               [1/3, 0.0, 0.0, 0.0],
                               [-1/3, 1.0, 0.0, 0.0],
                               [1.0, -1.0, 1.0, 0.0]]),
            "B": torch.Tensor([1/8, 3/8, 3/8, 1/8]),
            "C": torch.Tensor([0.0, 1/3, 2/3., 1.0])
        },
        "ralston": {
            "A": torch.Tensor([[0.0, 0.0, 0.0, 0.0],
                               [0.4, 0.0, 0.0, 0.0],
                               [0.29697761, 0.15875964, 0.0, 0.0],
                               [0.21810040, -3.05096516, 3.83286476, 0.0]]),
            "B": torch.Tensor([0.17476028, -0.55148066, 1.20553560, 0.17118478]),
            "C": torch.Tensor([0.0, 0.4, 0.45573725, 1.0])
        },
    }
}


@dataclass
class RKInversionOutput(BaseOutput):
    init_noise: torch.Tensor
    ori_image: Image.Image
    r: int
    method: str


class RKSolverFluxPipeline(FluxPipeline):
    @torch.no_grad()
    def inverse(
        self,
        image: Union[str, Image.Image],
        r: int = 4,
        method: str = "default",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        logger: Optional[Logger] = None,
        quiet: bool = False
    ):
        assert isinstance(self.scheduler, FlowMatchEulerDiscreteInversionScheduler)
        assert r in BUTCHER.keys()

        if isinstance(image, str):
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor
            im_loader = ImageLoader(logger, quiet)
            im_loader.load_image_from_path(image)
            im_loader.scale_image(match_long_size=height)
            image = im_loader.adjust_to_scale(scale_factor=16)
        elif isinstance(image, Image.Image):
            height = image.height
            width = image.width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = image2latent(
            self, 
            image, 
            generator=generator, 
            dtype=prompt_embeds.dtype, 
            device=device
        )
        latents = self._pack_latents(
            latents,
            batch_size,
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor)
        )
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu, is_inverse=True)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        if r == 1:
            desc = f"RK-Solver (r={r})"
            A = BUTCHER[r]["A"]
            B = BUTCHER[r]["B"]
            C = BUTCHER[r]["C"]
        else:
            if method == "default" or method not in BUTCHER[r]:
                method = BUTCHER[r]["default"]
            desc = f"RK-Solver (r={r}, {method})"
            A = BUTCHER[r][method]["A"]
            B = BUTCHER[r][method]["B"]
            C = BUTCHER[r][method]["C"]
        with EmojiProgressBar(total=num_inference_steps, desc=desc) as pbar:
            for t in timesteps:
                if self.interrupt:
                    continue

                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                h = sigma_next - sigma_cur

                K = []
                k_approx = torch.zeros_like(latents)
                for i in range(r):
                    sigma_use = sigma_cur + C[i] * h
                    t_use = sigma_use * self.scheduler.config.num_train_timesteps
                    latent_use = latents.clone()
                    for j in range(i):
                        latent_use += h * A[i][j] * K[j]

                    timestep = t_use.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latent_use,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    K.append(noise_pred)
                    k_approx += B[i] * noise_pred

                latents = latents + h * k_approx

                self.scheduler._step_index += 1
                pbar.update(1)
                
        # Offload all models
        self.maybe_free_model_hooks()

        return RKInversionOutput(init_noise=latents, ori_image=image, r=r, method=method)
    
    @torch.no_grad()
    def __call__(
        self,
        r: int = 4,
        method: str = "default",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        if r == 1:
            desc = f"RK-Solver (r={r})"
            A = BUTCHER[r]["A"]
            B = BUTCHER[r]["B"]
            C = BUTCHER[r]["C"]
        else:
            if method == "default" or method not in BUTCHER[r]:
                method = BUTCHER[r]["default"]
            desc = f"RK-Solver (r={r}, {method})"
            A = BUTCHER[r][method]["A"]
            B = BUTCHER[r][method]["B"]
            C = BUTCHER[r][method]["C"]
        with EmojiProgressBar(total=num_inference_steps, desc=desc) as pbar:
            for t in timesteps:
                if self.interrupt:
                    continue

                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                h = sigma_next - sigma_cur

                K = []
                k_approx = torch.zeros_like(latents)
                for i in range(r):
                    sigma_use = sigma_cur + C[i] * h
                    t_use = sigma_use * self.scheduler.config.num_train_timesteps
                    latent_use = latents.clone()
                    for j in range(i):
                        latent_use += h * A[i][j] * K[j]

                    timestep = t_use.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latent_use,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    K.append(noise_pred)
                    k_approx += B[i] * noise_pred

                latents = latents + h * k_approx
                
                self.scheduler._step_index += 1

                pbar.update(1)

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    @torch.no_grad()
    def edit(
        self,
        source_prompt: str,
        target_prompt: str,
        image: Union[str, Image.Image],
        r: int = 4,
        method: str = "default",
        hook_steps: Union[List[int], int] = None,
        hook_trans_block: bool = False,
        hook_single_trans_block: bool = True,
        hook_cc: bool = False,
        hook_ci: bool = False,
        hook_ic: bool = False,
        hook_ii: bool = False,
        hook_v: bool = False,
        cc_strategy: str = "replace",
        ci_strategy: str = "replace",
        ic_strategy: str = "replace",
        ii_strategy: str = "replace",
        vc_strategy: str = "replace",
        vi_strategy: str = "replace",
        cc_amplify: float = 1.0,
        ci_amplify: float = 1.0,
        ic_amplify: float = 1.0,
        vc_amplify: float = 1.0,
        extend_padding: bool = False,
        in_memory: bool = True,
        cache_dir: Optional[str] = None,
        remove_cache: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        inversion_guidance_scale: float = 1.0,
        denoise_guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        max_sequence_length: int = 512,
        logger: Optional[Logger] = None,
        quiet: bool = False
    ):
        assert source_prompt != target_prompt
        assert isinstance(source_prompt, str) and isinstance(target_prompt, str)
        assert isinstance(self.scheduler, FlowMatchEulerDiscreteInversionScheduler)
        assert r in BUTCHER.keys()

        if isinstance(image, str):
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor
            im_loader = ImageLoader(logger, quiet)
            im_loader.load_image_from_path(image)
            im_loader.scale_image(match_long_size=height)
            image = im_loader.adjust_to_scale(scale_factor=16)
        elif isinstance(image, Image.Image):
            height = image.height
            width = image.width

        batch_size = 1
        device = self._execution_device

        # encode source prompt
        source_prompt_embeds, source_pooled_prompt_embeds, source_text_ids = self.encode_prompt(
            prompt=source_prompt.replace("[", "").replace("]", ""),
            prompt_2=None,
            device=device,
            max_sequence_length=max_sequence_length
        )
        
        # prepare latent variables for inversion
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = image2latent(
            self, 
            image, 
            generator=generator, 
            dtype=source_prompt_embeds.dtype, 
            device=device
        )
        latents = self._pack_latents(
            latents,
            batch_size,
            num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width) // self.vae_scale_factor)
        )
        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            source_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # prepare timesteps for inversion
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu, is_inverse=True)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], inversion_guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # prepare attention hook
        attn_hook = DDTA(
            self,
            hook_trans_block=hook_trans_block,
            hook_single_trans_block=hook_single_trans_block,
            hook_cc=hook_cc,
            hook_ci=hook_ci,
            hook_ic=hook_ic,
            hook_ii=hook_ii,
            hook_v=hook_v,
            cc_strategy=cc_strategy,
            ci_strategy=ci_strategy,
            ic_strategy=ic_strategy,
            ii_strategy=ii_strategy,
            vc_strategy=vc_strategy,
            vi_strategy=vi_strategy,
            cc_amplify=cc_amplify,
            ci_amplify=ci_amplify,
            ic_amplify=ic_amplify,
            vc_amplify=vc_amplify,
            extend_padding=extend_padding,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            in_memory=in_memory,
            cache_dir=cache_dir,
            remove_cache=remove_cache
        )
        attn_hook.set_hook_steps(steps=hook_steps, inverse=True, order=r)
        attn_hook.set_edit_prompt(source_prompt, target_prompt)
        attn_hook.store_mode = True
        attn_hook.activate()

        # inversion
        if r == 1:
            desc = f"RK-Solver (r={r})"
            A = BUTCHER[r]["A"]
            B = BUTCHER[r]["B"]
            C = BUTCHER[r]["C"]
        else:
            if method == "default" or method not in BUTCHER[r]:
                method = BUTCHER[r]["default"]
            desc = f"RK-Solver (r={r}, {method})"
            A = BUTCHER[r][method]["A"]
            B = BUTCHER[r][method]["B"]
            C = BUTCHER[r][method]["C"]
        with EmojiProgressBar(total=num_inference_steps, desc=desc) as pbar:
            for t in timesteps:

                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                h = sigma_next - sigma_cur

                K = []
                k_approx = torch.zeros_like(latents)
                for i in range(r):
                    sigma_use = sigma_cur + C[i] * h
                    t_use = sigma_use * self.scheduler.config.num_train_timesteps
                    latent_use = latents.clone()
                    for j in range(i):
                        latent_use += h * A[i][j] * K[j]

                    timestep = t_use.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latent_use,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=source_pooled_prompt_embeds,
                        encoder_hidden_states=source_prompt_embeds,
                        txt_ids=source_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    attn_hook.next_step(True)
                    K.append(noise_pred)
                    k_approx += B[i] * noise_pred

                latents = latents + h * k_approx
                self.scheduler._step_index += 1
                
                pbar.update(1)

        # encode target prompt
        target_prompt_embeds, target_pooled_prompt_embeds, target_text_ids = self.encode_prompt(
            prompt=target_prompt.replace("[", "").replace("]", ""),
            prompt_2=None,
            device=device,
            max_sequence_length=max_sequence_length
        )
        
        # prepare latent variables for denoising
        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            target_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # prepare timesteps for denoising
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu, is_inverse=False)
        timesteps = self.scheduler.timesteps

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], denoise_guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # set attention hook for denoising
        attn_hook.set_hook_steps(steps=hook_steps, inverse=False, order=r)
        attn_hook.store_mode = False

        # denoising loop
        with EmojiProgressBar(total=num_inference_steps, desc=desc) as pbar:
            for t in timesteps:
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
                sigma_next = self.scheduler.sigmas[self.scheduler.step_index+1]
                h = sigma_next - sigma_cur

                K = []
                k_approx = torch.zeros_like(latents)
                for i in range(r):
                    sigma_use = sigma_cur + C[i] * h
                    t_use = sigma_use * self.scheduler.config.num_train_timesteps
                    latent_use = latents.clone()
                    for j in range(i):
                        latent_use += h * A[i][j] * K[j]

                    timestep = t_use.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latent_use,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=target_pooled_prompt_embeds,
                        encoder_hidden_states=target_prompt_embeds,
                        txt_ids=target_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    attn_hook.next_step(False)
                    K.append(noise_pred)
                    k_approx += B[i] * noise_pred

                latents = latents + h * k_approx
                self.scheduler._step_index += 1

                pbar.update(1)
        
        # output
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        edit_image = self.vae.decode(latents, return_dict=False)[0]
        edit_image = self.image_processor.postprocess(edit_image, output_type=output_type)
        
        # remove attention hook
        attn_hook.remove()

        # offload all models
        self.maybe_free_model_hooks()

        return EditOutput(ori_image=image, edit_image=edit_image[0])
