import os.path as osp
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from diffusers.utils import BaseOutput
from PIL import Image
from mlcbase import Logger


@dataclass
class InversionOutput(BaseOutput):
    init_noise: torch.Tensor
    ori_image: Image.Image


@dataclass
class EditOutput(BaseOutput):
    ori_image: Image.Image
    edit_image: Image.Image


class ImageLoader:
    def __init__(self, logger: Optional[Logger] = None, quiet: bool = False):
        if logger is None:
            logger = Logger()
            logger.init_logger()
        if quiet:
            logger.set_quiet()
        else:
            logger.set_activate()
        self.logger = logger

        self._image = None
        self._path = None
        self._mode = None

    @property
    def image(self):
        if self._image is None:
            self.logger.error("Image not loaded yet.")
            raise ValueError("Image not loaded yet.")
        return self._image

    @property
    def path(self):
        return self._path
    
    def set_image(self, image: Image.Image):
        assert isinstance(image, Image.Image), "image must be a PIL.Image object"
        self._image = image
        self._path = "manual set image"

    def load_image_from_path(self, path: str, color_mode: str = "RGB"):
        self._image = Image.open(path).convert(color_mode)
        self._path = path
        self._mode = color_mode
        self.logger.info(f"[Load] {osp.basename(path)} | original size: {self.image.size}")
        return self.image
    
    def direct_resize_image(self, size: Sequence[int]):
        assert len(size) == 2, "size must be a sequence of two integers"
        self._image = self._image.resize(size)
        self.logger.info(f"[Direct Resize] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image
    
    def scale_image(
        self, 
        match_long_size: Optional[int] = None, 
        match_short_size: Optional[int] = None, 
        scale_ratio: Optional[float] = None
    ):
        assert match_long_size is None or match_short_size is None or scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"

        w, h = self.image.size

        if match_long_size is not None:
            assert match_short_size is None and scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            if w > h:
                ratio = match_long_size / w
            else:
                ratio = match_long_size / h
        
        if match_short_size is not None:
            assert match_long_size is None and scale_ratio is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            if w > h:
                ratio = match_short_size / h
            else:
                ratio = match_short_size / w

        if scale_ratio is not None:
            assert match_long_size is None and match_short_size is None, "match_long_size, match_short_size and scale_ratio cannot be provided at the same time"
            ratio = scale_ratio

        self._image = self._image.resize((int(w * ratio), int(h * ratio)))
        self.logger.info(f"[Scaling] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image
    
    def adjust_to_scale(
        self,
        scale_factor: int = 16,
        method: str = "center_crop",
        offset: Optional[Sequence[int]] = None,
        resize_round_up: bool = False
    ):
        assert method in ["center_crop", "offset", "resize"], "method must be one of ['center_crop', 'offset', 'resize']"
        if method == "offset":
            assert offset is not None, "offset must be provided when method is 'offset'"
            assert len(offset) == 4, "offset must be a sequence of four integers"

        w, h = self.image.size
        if w % scale_factor == 0 and h % scale_factor == 0:
            return self.image
        
        image = np.array(self.image)

        if h % scale_factor != 0:
            if method == "resize":
                new_h = h - h % scale_factor
                if resize_round_up:
                    new_h += scale_factor
                image = np.array(Image.fromarray(image, self._mode).resize((w, new_h)))
            
            if method == "center_crop":
                start = (h % scale_factor) // 2
                end = h - (h % scale_factor - start)
                if len(image.shape) == 3:
                    image = image[start:end, :, :]
                else:
                    image = image[start:end]

            if method == "offset":
                assert offset[1] + offset[3] == (h % scale_factor)
                if len(image.shape) == 3:
                    image = image[offset[1]:h-offset[3], :, :]
                else:
                    image = image[offset[1]:h-offset[3]]
        h = image.shape[0]

        if w % scale_factor != 0:
            if method == "resize":
                new_w = w - w % scale_factor
                if resize_round_up:
                    new_w += scale_factor
                image = np.array(Image.fromarray(image, self._mode).resize((new_w, h)))
            
            if method == "center_crop":
                start = (w % scale_factor) // 2
                end = w - (w % scale_factor - start)
                if len(image.shape) == 3:
                    image = image[:, start:end, :]
                else:
                    image = image[:, start:end]
            
            if method == "offset":
                assert offset[0] + offset[2] == (w % scale_factor)
                if len(image.shape) == 3:
                    image = image[:, offset[0]:w-offset[2], :]
                else:
                    image = image[:, offset[0]:w-offset[2]]

        self._image = Image.fromarray(image, self._mode)
        self.logger.info(f"[Adjust2Scale] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image
    
    def pad_to_scale(
        self,
        scale_factor: int = 128,
        pad_value: int = 0
    ):
        assert 0 <= pad_value <= 255, "pad_value must be in the range [0, 255]"

        w, h = self.image.size

        new_h = (h + scale_factor - 1) // scale_factor * scale_factor
        new_w = (w + scale_factor - 1) // scale_factor * scale_factor
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top

        image = np.array(self.image)
        if len(image.shape) == 3:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=pad_value,
            )

        self._image = Image.fromarray(image, self._mode)
        self.logger.info(f"[Pad2Scale] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image

    def center_crop_to_square(self):
        w, h = self.image.size

        image = np.array(self.image)
        diff = abs(w - h)
        if w > h:
            start = diff // 2
            end = w - (diff - start)
            if len(image.shape) == 3:
                image = image[:, start:end, :]
            else:
                image = image[:, start:end]
        else:
            start = diff // 2
            end = h - (diff - start)
            if len(image.shape) == 3:
                image = image[start:end, :, :]
            else:
                image = image[start:end]

        self._image = Image.fromarray(image, self._mode)
        self.logger.info(f"[CenterCrop2Square] {osp.basename(self.path)} | current size: {self.image.size}")
        return self.image


def image2latent(
    pipe, 
    image, 
    generator: Optional[torch.Generator] = None,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None, 
    device: Optional[str] = None, 
):
    if dtype is None:
        dtype = pipe.transformer.dtype
    if device is None:
        device = pipe._execution_device
    
    image = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
    image = image.to(dtype=dtype, device=device)

    if requires_grad:
        latents = pipe.vae.encode(image).latent_dist.sample(generator)
    else:
        with torch.no_grad():
            latents = pipe.vae.encode(image).latent_dist.sample(generator)

    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    
    return latents


def latent2image(
    pipe, 
    latents, 
    output_type="pil", 
    requires_grad: bool = False
):
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    if requires_grad:
        image = pipe.vae.decode(latents, return_dict=False)[0]
    else:
        with torch.no_grad():
            image = pipe.vae.decode(latents, return_dict=False)[0]

    image = pipe.image_processor.postprocess(image, output_type=output_type)
    if output_type == "pil":
        image = image[0]
        
    return image


def pil2tensor(image: Image.Image, normalize: bool = False) -> torch.Tensor:
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    if normalize:
        image = 2.0 * image - 1.0  # normalize to [-1, 1]
    return image
