"""
OmniGen2 Diffusion Pipeline

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers.models.autoencoders import AutoencoderKL
from ...models.transformers import OmniGen2Transformer2DModel
from ...models.transformers.repo import OmniGen2RotaryPosEmbed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass

import PIL.Image

from diffusers.utils import BaseOutput

from omnirecolor.pipelines.image_processor import OmniGen2ImageProcessor

from omnirecolor.utils.teacache_util import TeaCacheParams

from ..lora_pipeline import OmniGen2LoraLoaderMixin
from torchvision.transforms import ToPILImage


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from ...cache_functions import cache_init 
import gc
import matplotlib.pyplot as plt
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FMPipelineOutput(BaseOutput):
    """
    Output class for OmniGen2 pipeline.

    Args:
        images (Union[List[PIL.Image.Image], np.ndarray]): 
            List of denoised PIL images of length `batch_size` or numpy array of shape 
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGen2Pipeline(DiffusionPipeline, OmniGen2LoraLoaderMixin):
    """
    Pipeline for text-to-image generation using OmniGen2.

    This pipeline implements a text-to-image generation model that uses:
    - Qwen2.5-VL for text encoding
    - A custom transformer architecture for image generation
    - VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler for noise scheduling

    Args:
        transformer (OmniGen2Transformer2DModel): The transformer model for image generation.
        vae (AutoencoderKL): The VAE model for image encoding/decoding.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for noise scheduling.
        text_encoder (Qwen2_5_VLModel): The text encoder model.
        tokenizer (Union[Qwen2Tokenizer, Qwen2TokenizerFast]): The tokenizer for text processing.
    """

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: OmniGen2Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen2_5_VLForConditionalGeneration,
        processor,
    ) -> None:
        """
        Initialize the OmniGen2 pipeline.

        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: Union[List[PIL.Image.Image], PIL.Image.Image],
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Optional[torch.FloatTensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """
        if batch_size == 1:
            images = [images]
        latents = []
        for i, img in enumerate(images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    img_j = self.image_processor.preprocess(img_j, max_pixels=max_pixels, max_side_length=max_side_length)
                    if j==0:
                        target_size = img_j.shape[2:]  # (H, W)输入图像的尺寸
                    if j==1 and self.input_type=='palette':
                        # Resize palette 图像到参考图一致大小
                        img_j = F.interpolate(img_j, size=target_size, mode='bilinear',
                                                        align_corners=False)
                    ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
            else:
                ref_latents = None
            for _ in range(num_images_per_prompt):
                latents.append(ref_latents)

        return latents
    
    def _get_qwen2_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from the Qwen2 text encoder.

        Args:
            prompt: The prompt or list of prompts to encode.
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The prompt embeddings tensor
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # text_inputs = self.processor.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=max_sequence_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Gemma can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask
    
    def _apply_chat_template(self, prompt: str):
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates high-quality images based on user instructions.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return prompt

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self._apply_chat_template(_prompt) for _prompt in prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [self._apply_chat_template(_negative_prompt) for _negative_prompt in negative_prompt]

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size * num_images_per_prompt, -1
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale
    
    @property
    def cfg_range(self):
        return self._cfg_range
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        palette_list: Optional[List[List[int]]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if 'palette' in prompt:
            self.input_type='palette'
            palette_rgb=palette_list
        elif 'reference' in prompt:
            self.input_type='reference'
        else:
            self.input_type = None

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            device=device,
            dtype=dtype,
        )

        if input_images is None:
            input_images = []

        if len(input_images) != 0 and align_res:
            width, height = ref_latents[0][0].shape[-1] * self.vae_scale_factor, ref_latents[0][0].shape[-2] * self.vae_scale_factor
            ori_width, ori_height = width, height
        # if len(input_images) == 1 and align_res:
        #     width, height = ref_latents[0][0].shape[-1] * self.vae_scale_factor, ref_latents[0][0].shape[-2] * self.vae_scale_factor
        #     ori_width, ori_height = width, height
        # else:
        #     ori_width, ori_height = width, height
        #
        #     cur_pixels = height * width
        #     ratio = (max_pixels / cur_pixels) ** 0.5
        #     ratio = min(ratio, 1.0)
        #
        #     height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16
        
        if len(input_images) == 0:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )
        if "reference" in prompt:
            image = self.ref_processing(
                latents=latents,
                ref_latents=ref_latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                device=device,
                dtype=dtype,
                verbose=verbose,
                step_func=step_func,
                sw_reference=input_images[1],
                sw_debug=False,
                width=ori_width,
                height=ori_height
            )
        elif "palette" in prompt:
            image = self.pal_processing(
                latents=latents,
                ref_latents=ref_latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                device=device,
                dtype=dtype,
                verbose=verbose,
                step_func=step_func,
                pal_list=palette_rgb,
                sw_debug=False,
                width=ori_width,
                height=ori_height
            )
        else:
            image = self.processing(
                latents=latents,
                ref_latents=ref_latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                device=device,
                dtype=dtype,
                verbose=verbose,
                step_func=step_func,
            )

        image = F.interpolate(image, size=(ori_height, ori_width), mode='bilinear')

        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)

    def processing(
            self,
            latents,
            ref_latents,
            prompt_embeds,
            freqs_cis,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            num_inference_steps,
            timesteps,
            device,
            dtype,
            verbose,
            step_func=None,
    ):
        batch_size = latents.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif self.transformer.enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= \
                                                                  self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= \
                                                                    self.cfg_range[1] else 1.0

                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                                 text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

    def sliced_wasserstein(self,loss, pixels_gen, pixels_ref, num_slices=10, use_differenatable_historgam_matching=False):
        for slice_idx in range(num_slices):
            R_trans = rand_rotation_matrix(deflection=1.0)
            R_trans = torch.Tensor(R_trans)
            R_trans = R_trans.to(self.device).to(torch.bfloat16)
            pixels_gen_rotated = pixels_gen.T @ R_trans
            pixels_ref_rotated = pixels_ref.T @ R_trans
            for dim_idx in range(3):
                x_samples_3d_rotated_slice = pixels_gen_rotated[:, dim_idx]
                y_samples_3d_rotated_slice = pixels_ref_rotated[:, dim_idx]

                if not use_differenatable_historgam_matching:
                    rand_idxes_len = min(len(x_samples_3d_rotated_slice), len(y_samples_3d_rotated_slice))
                    rand_idxes = np.random.randint(0, rand_idxes_len, rand_idxes_len)
                    x_samples_3d_rotated_slice = x_samples_3d_rotated_slice[rand_idxes]
                    y_samples_3d_rotated_slice = y_samples_3d_rotated_slice[rand_idxes]

                    x_samples_3d_rotated_slice = torch.sort(x_samples_3d_rotated_slice).values
                    y_samples_3d_rotated_slice = torch.sort(y_samples_3d_rotated_slice).values
                    loss += torch.mean(torch.abs(x_samples_3d_rotated_slice - y_samples_3d_rotated_slice))
                else:
                    min_range = min(
                        x_samples_3d_rotated_slice.min().item(),
                        y_samples_3d_rotated_slice.min().item()
                    )
                    max_range = max(
                        x_samples_3d_rotated_slice.max().item(),
                        y_samples_3d_rotated_slice.max().item()
                    )
                    grid_size = 400
                    full_range = torch.linspace(min_range - 0.05, max_range + 0.05, grid_size)
                    full_range = full_range.to(self.device)
                    x_cdf = get_cdf(x_samples_3d_rotated_slice, full_range)
                    y_cdf = get_cdf(y_samples_3d_rotated_slice, full_range)
                    loss += torch.mean(torch.abs(x_cdf - y_cdf))
        return loss / num_slices
    def do_sw_guidance(self, sw_steps, sw_u_lr, latents, t,
                       prompt_embeds,freqs_cis,prompt_attention_mask,ref_latents,
                       pixels_ref):


        u = torch.zeros_like(latents, requires_grad=True, dtype=torch.bfloat16)
        if self.sw_debug:
            print(
                f"t {t}, prompt_embeds {prompt_embeds.shape}, {prompt_embeds.dtype} grad: {prompt_embeds.requires_grad}")
            print(f"latents: {latents.shape}, {latents.dtype} grad: {latents.requires_grad}")

        for tt in range(sw_steps):
            u = u.detach()
            u.requires_grad = True
            print(latents.dtype, u.dtype)
            x_hat_t = latents.detach() + u
            # print("\n===== GRADITY CHECK =====")
            # print("u.requires_grad:", u.requires_grad)
            # print("u.grad_fn:", u.grad_fn)
            #
            # print("x_hat_t.requires_grad:", x_hat_t.requires_grad)
            # print("x_hat_t.grad_fn:", x_hat_t.grad_fn)
            # print("=========================\n")
            # with torch.no_grad():
            model_pred = self.predict(
                t=t,
                latents=x_hat_t,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                prompt_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=ref_latents,
            )
            # sch_step = self.scheduler.step(model_pred, t, x_hat_t, return_dict=False)[0]
            sch_step=euler_step_no_state(self.scheduler,model_pred,t,x_hat_t)
            x_hat_1=sch_step+(1-t)*model_pred

            # ------------ Compute loss
            image = self.vae.decode(x_hat_1 / self.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            pixels_gen = image.squeeze(0).reshape(3, -1)
            # indx = np.random.randint(0, len(pixels_gen.T), 10_000)
            # gen_mean = torch.mean(pixels_gen[:, indx], dim=1)
            # ref_mean = torch.mean(pixels_ref[:, indx], dim=1).to(self.device)
            # gen_cov = torch.cov(pixels_gen[:, indx])
            # ref_cov = torch.cov(pixels_ref[:, indx]).to(self.device)

            loss = 0
            # loss = torch.mean(torch.abs(gen_mean - ref_mean)) + torch.mean(torch.abs(gen_cov - ref_cov))
            loss = self.sliced_wasserstein(loss, pixels_gen, pixels_ref)
            if self.sw_debug:
                print(f"tt{tt}  loss ", loss)

            u_t_grad = torch.autograd.grad(loss, u)[0]
            if self.sw_debug:
                print(f"tt{tt}  u_t_grad.std() ", u_t_grad.std())
            with torch.no_grad():
                u_t_grad = u_t_grad  # / u_t_grad.std()
                u.data = u.data - sw_u_lr * u_t_grad
            # if self.sw_debug and (t % 2) and (tt == 0):
            if self.sw_debug and (tt == 0):
                # 解码 latent
                pil_image_tensor = self.vae.decode(x_hat_1 / self.vae.config.scaling_factor).sample

                # 转 float32 并归一化到 [0,1]
                pil_image_tensor = pil_image_tensor.to(torch.float32)
                pil_image_tensor = (pil_image_tensor / 2 + 0.5).clamp(0, 1)

                # 去掉 batch 维度
                pil_image_tensor = pil_image_tensor.squeeze(0)  # shape (C,H,W)

                # 转为 numpy array 并转为 PIL
                array = (pil_image_tensor.detach().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                from PIL import Image
                pil_image = Image.fromarray(array)

                # 保存
                save_path = f"/home/ww/work/OmniGen2-main/sw_image/2_sw_debug_t{self.scheduler.step_index}_tt{tt}.png"
                pil_image.save(save_path)
                print(f"[SW DEBUG] Saved image to {save_path}")

                # 释放显存
                del pil_image_tensor, array, pil_image
                torch.cuda.empty_cache()

        with torch.no_grad():
            latents = latents.detach() + u.detach()
        del model_pred, sch_step,
        del x_hat_1, image, pixels_gen,
        del loss, u_t_grad,
        # del gen_cov, ref_cov
        # del loss, u_t_grad, gen_mean, ref_mean,
        gc.collect()
        torch.cuda.empty_cache()
        return latents

    def ref_processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        verbose,
        step_func=None,
        sw_steps: int = 1,
        # sw_u_lr: float = 0.05 * 10 ** 3,
        #sw_u_lr: float = 5,#for pal
        sw_u_lr: float = 50,  # for ref
        sw_reference: PIL.Image = None,
        sw_debug: bool = False,
        num_guided_steps: int = 20,
        width: int = None,
        height: int = None,
    ):
        self.sw_debug = sw_debug
        if sw_reference is not None:
            ref_im = sw_reference.resize((width, height))
            self.ref_im = ref_im
            self.num_guided_steps = num_guided_steps
            pixels_ref = torch.Tensor(np.array(ref_im) / 255).permute(2, 1, 0).reshape(3, -1).to(device).to(
                torch.bfloat16)
            # self.vae.encoder.cpu()


        batch_size = latents.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif self.transformer.enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.scheduler.step_index and self.scheduler.step_index > num_inference_steps - 1:
                    print("Stopping early because scheduler already reached end.")
                    break
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                if sw_reference is not None:
                    if self.num_guided_steps is None or i < self.num_guided_steps:
                        # print("before_sw_step_index:", self.scheduler.step_index, "i:", i)

                        t_now=t
                        with torch.enable_grad():
                            latents = self.do_sw_guidance(sw_steps, sw_u_lr, latents, t_now,
                                                          prompt_embeds,freqs_cis,prompt_attention_mask,ref_latents,
                                                          pixels_ref)
                        # print("after_sw_step_index:", self.scheduler.step_index, "i:", i)
                        # if i == len(timesteps) - 1 or (
                        #         (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        #     progress_bar.update()
                        #
                        # if step_func is not None:
                        #     step_func(i, self._num_timesteps)

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0

                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                        text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)
                # print("before_sch_step_index:", self.scheduler.step_index, "i:", i)
                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
                # print("after_sch_step_index:", self.scheduler.step_index, "i:", i)
                latents = latents.to(dtype=dtype)

                pil_image_tensor = self.vae.decode(latents / self.vae.config.scaling_factor).sample

                # 转 float32 并归一化到 [0,1]
                pil_image_tensor = pil_image_tensor.to(torch.float32)
                pil_image_tensor = (pil_image_tensor / 2 + 0.5).clamp(0, 1)

                # 去掉 batch 维度
                pil_image_tensor = pil_image_tensor.squeeze(0)  # shape (C,H,W)

                # 转为 numpy array 并转为 PIL
                array = (pil_image_tensor.detach().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                from PIL import Image
                pil_image = Image.fromarray(array)
                if self.sw_debug:
                    # 保存
                    save_path = f"/home/ww/work/OmniGen2-main/sw_image/1_sw_debug_t{self.scheduler.step_index}.png"
                    pil_image.save(save_path)
                    print(f"[SW DEBUG] Saved image to {save_path}")
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)


        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        
        return image

    def pal_processing(
            self,
            latents,
            ref_latents,
            prompt_embeds,
            freqs_cis,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            num_inference_steps,
            timesteps,
            device,
            dtype,
            verbose,
            step_func=None,
            sw_steps: int = 1,
            # sw_u_lr: float = 0.05 * 10 ** 3,
            sw_u_lr: float = 30,
            pal_list: List = None,
            sw_debug: bool = False,
            num_guided_steps: int = 20,
            width: int = None,
            height: int = None,
    ):
        self.sw_debug = sw_debug
        if pal_list is not None:
            self.num_guided_steps = num_guided_steps
            pixels_ref = palette_to_pixels_ref(pal_list, device=self.device, dtype=torch.bfloat16)

        batch_size = latents.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif self.transformer.enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.scheduler.step_index and self.scheduler.step_index > num_inference_steps - 1:
                    print("Stopping early because scheduler already reached end.")
                    break
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                if pal_list is not None:
                    if self.num_guided_steps is None or i < self.num_guided_steps:
                        # print("before_sw_step_index:", self.scheduler.step_index, "i:", i)

                        t_now = t
                        with torch.enable_grad():
                            latents = self.do_sw_guidance(sw_steps, sw_u_lr, latents, t_now,
                                                          prompt_embeds, freqs_cis, prompt_attention_mask, ref_latents,
                                                          pixels_ref)
                        # print("after_sw_step_index:", self.scheduler.step_index, "i:", i)
                        # if i == len(timesteps) - 1 or (
                        #         (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        #     progress_bar.update()
                        #
                        # if step_func is not None:
                        #     step_func(i, self._num_timesteps)

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= \
                                                                  self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= \
                                                                    self.cfg_range[1] else 1.0

                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                                 text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)
                # print("before_sch_step_index:", self.scheduler.step_index, "i:", i)
                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
                # print("after_sch_step_index:", self.scheduler.step_index, "i:", i)
                latents = latents.to(dtype=dtype)

                pil_image_tensor = self.vae.decode(latents / self.vae.config.scaling_factor).sample

                # 转 float32 并归一化到 [0,1]
                pil_image_tensor = pil_image_tensor.to(torch.float32)
                pil_image_tensor = (pil_image_tensor / 2 + 0.5).clamp(0, 1)

                # 去掉 batch 维度
                pil_image_tensor = pil_image_tensor.squeeze(0)  # shape (C,H,W)

                # 转为 numpy array 并转为 PIL
                array = (pil_image_tensor.detach().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                from PIL import Image
                pil_image = Image.fromarray(array)
                if self.sw_debug:
                    # 保存
                    save_path = f"/home/ww/work/OmniGen2-main/sw_image/1_sw_debug_t{self.scheduler.step_index}.png"
                    pil_image.save(save_path)
                    print(f"[SW DEBUG] Saved image to {save_path}")
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if step_func is not None:
                    step_func(i, self._num_timesteps)

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image
    def predict(
        self,
        t,
        latents,
        prompt_embeds,
        freqs_cis,
        prompt_attention_mask,
        ref_image_hidden_states,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape
        
        optional_kwargs = {}
        if 'ref_image_hidden_states' in set(inspect.signature(self.transformer.forward).parameters.keys()):
            optional_kwargs['ref_image_hidden_states'] = ref_image_hidden_states
        
        model_pred = self.transformer(
            latents,
            timestep,
            prompt_embeds,
            freqs_cis,
            prompt_attention_mask,
            **optional_kwargs
        )
        return model_pred


def print_images(pil_imgs, with_density=False, points=200, titles=None, s=10):
    fig = plt.figure(figsize=(20, 10))
    n = len(pil_imgs)
    azim = 236  # np.random.uniform(0,360)
    nrows = 2 if with_density else 1
    for i in range(n):
        img = pil_imgs[i]
        ax = fig.add_subplot(nrows, n, i + 1)
        ax.imshow(img)
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')

        if with_density:
            img = img.convert('RGB')
            w, h = img.size
            arr = np.array(img).reshape(w * h, 3) / 255
            idx = np.random.permutation(w * h)[:points]

            ax = fig.add_subplot(nrows, n, n + i + 1, projection='3d')
            r = arr[idx, 0]
            g = arr[idx, 1]
            b = arr[idx, 2]
            ax.scatter(r, g, b, c=arr[idx, :], alpha=0.5, s=s)
            ax.view_init(elev=30, azim=azim)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_zaxis().set_ticklabels([])
    return fig


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
def sigmoid(x):
    return 1 - 1 / (1 + torch.exp(-35*x))
    #return x<0
def get_cdf(x_samples, full_range):
    return sigmoid((x_samples[:,None]-full_range[None,:])).sum(0)/len(x_samples)


def euler_step_no_state(scheduler, model_output, timestep, sample):
    """
    Compute the next latent without touching scheduler.step_index.

    Args:
        scheduler: FlowMatchEulerDiscreteScheduler
        model_output: predicted derivative (same as model_pred)
        timestep: current t in scheduler.timesteps (Tensor or float)
        sample: current latent x_t

    Returns:
        next_sample: the predicted x_{t_next}
    """
    # Convert to float32 to match scheduler behavior
    sample_32 = sample.to(torch.float32)

    # Find current index in scheduler.timesteps (avoid using scheduler.step_index)
    # scheduler.timesteps is a tensor like tensor([999, 980, ..., 0])
    timesteps = scheduler.timesteps

    # Find index of t
    # 如果 timestep 是 Tensor，就 detach 再处理
    if isinstance(timestep, torch.Tensor):
        t_val = timestep.item()
    else:
        t_val = timestep

    # 找出 t 的位置
    # (scheduler.timesteps 可能是降序的)
    idx = (timesteps == t_val).nonzero(as_tuple=True)[0].item()

    # 如果已经是最后一步，直接返回 sample
    if idx == len(timesteps) - 1:
        return sample

    # 获取 t 和 t_next
    t_cur = timesteps[idx]
    t_next = timesteps[idx + 1]

    # Euler update: x_{t_next} = x_t + (t_next - t) * f(x_t)
    next_sample = sample_32 + (t_next - t_cur) * model_output

    return next_sample.to(model_output.dtype)
import torch
import numpy as np
import cv2  # 用于 RGB <-> LAB

def rgb_to_lab_torch(rgb_tensor):
    """
    输入: rgb_tensor (N, 3) [0,1]
    输出: lab_tensor (N, 3)
    使用 cv2 进行转换 (CIE Lab)，保持与 skimage 保持一致
    """
    rgb_np = rgb_tensor.detach().cpu().numpy().reshape(-1, 1, 3).astype(np.float32)
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab_tensor = torch.from_numpy(lab_np).to(rgb_tensor.device, dtype=torch.float32)
    return lab_tensor


def palette_to_pixels_ref(
    palette_rgb,
    n_per_color=2000,
    device='cuda',
    dtype=torch.float32,
    color_space='rgb',   # 'rgb' 或 'lab'
    sigma=2.0            # RGB 噪声会自动缩小
):
    """
    将 K×3 的调色板颜色列表转换为 (3, N) 的像素分布，用于 sliced_wasserstein 引导

    输入:
        palette_rgb: list 或 numpy 数组, shape (K, 3)
            - RGB 颜色, 范围 [0-255] 或 [0-1]
        n_per_color: int
            - 每个颜色采样多少像素
        device: 'cuda' 或 'cpu'
        dtype: torch.float32 或 torch.bfloat16
        color_space: 'rgb' 或 'lab'
            - 是否转为 Lab 后加噪声
        sigma: float
            - 噪声强度 (Lab 下默认 2, RGB 下会自动除以 255)

    输出:
        pixels_ref: torch.Tensor, shape = (3, N), 范围 [0,1]
            - sliced_wasserstein(pixels_gen, pixels_ref) 直接适配
    """

    # --- Load palette ---
    pal = torch.tensor(palette_rgb, dtype=torch.float32, device=device)
    if pal.max() > 1.0:
        pal = pal / 255.0   # normalize to [0,1]

    K = pal.shape[0]

    # --- Repeat ---
    samples = pal.unsqueeze(1).repeat(1, n_per_color, 1).reshape(-1, 3)   # (K*n, 3)

    # --- Convert to Lab + add noise ---
    if color_space.lower() == 'lab':
        samples_lab = rgb_to_lab_torch(samples)
        noise = torch.randn_like(samples_lab) * sigma
        samples = samples_lab + noise

        # Lab 不 clamp 到 [0,1]，保持原始 Lab 空间动态范围
        # 但你要与 RGB pixels_gen 对齐，所以 decode 后必须转 RGB 再比较
        # 因此这里建议最终仍转回 RGB
        samples = samples.clamp(0, 255)
        samples = samples / 255.0

    else:  # RGB noise
        noise = torch.randn_like(samples) * (sigma / 255.0)
        samples = samples + noise
        samples = samples.clamp(0.0, 1.0)

    # --- Return as (3,N) ---
    pixels_ref = samples.T.contiguous()

    return pixels_ref.to(device=device, dtype=dtype)
