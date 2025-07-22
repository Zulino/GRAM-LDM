# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan

# from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from ..models.unet import UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_available, logging, replace_example_docstring
# from diffusers.utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
from .torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline

from .imagebind_data import load_and_transform_audio_data_from_waveform, load_and_transform_video_data, load_and_transform_text, load_and_transform_vision_data, waveform2melspec
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType

import torchaudio

import soundfile as sf
import os

from utils.utils_for_fast_inference import build_batch

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AudioLDMPipeline

        >>> pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "A hammer hitting a wooden surface"
        >>> audio = pipe(prompt).audio[0]
        ```
"""

class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000, n_mels=128, n_fft=1024, hop_length=250
    ):
        super().__init__()
        self.waveform_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, 
                        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.waveform_to_mel(waveform)

        return mel


class AudioLDMPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode audios to and from latent representations.
        text_encoder ([`ClapTextModelWithProjection`]):
            Frozen text-encoder. AudioLDM uses the text portion of
            [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap#transformers.ClapTextModelWithProjection),
            specifically the [RoBERTa HSTAT-unfused](https://huggingface.co/laion/clap-htsat-unfused) variant.
        tokenizer ([`PreTrainedTokenizer`]):
            Tokenizer of class
            [RobertaTokenizer](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer).
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded audio latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        vocoder ([`SpeechT5HifiGan`]):
            Vocoder of class
            [SpeechT5HifiGan](https://huggingface.co/docs/transformers/main/en/model_doc/speecht5#transformers.SpeechT5HifiGan).
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapTextModelWithProjection,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)


    # def volume_computation(self, anchor, *inputs):
    #     """
    #     General function to compute volume for contrastive learning loss functions.
    #     Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    #     Args:
    #     - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    #     - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    #     Returns:
    #     - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    #     """
    #     # Ensure all inputs are on the same device as the anchor
    #     device = anchor.device
    #     inputs = [input_tensor.to(device) for input_tensor in inputs]

    #     batch_size1 = anchor.shape[0]
    #     # Assuming all inputs in *inputs have the same batch_size as the first one for this calculation
    #     # This is consistent with how GRAM loss is typically applied (all modalities aligned for the batch)
    #     batch_size2 = inputs[0].shape[0] 
    #     if batch_size1 != batch_size2:
    #         # This case needs careful handling if inputs can have different batch sizes than anchor
    #         # For typical contrastive learning, batch_size1 should be equal to batch_size2
    #         # If num_waveforms_per_prompt > 1, anchor (e.g. text) might need repeating to match inputs (e.g. audio)
    #         # The calling function should handle this alignment.
    #         # Here, we'll proceed assuming the calling code has ensured batch_size1 == batch_size2 for all inputs.
    #         # If they are not, the einsum and expand operations might behave unexpectedly or error out.
    #         # For the GRAM loss as described, batch_size1 (from anchor) and batch_size2 (from inputs[0])
    #         # are expected to be the effective batch size for the contrastive comparison.
    #          pass # Let it proceed, caller must ensure dimensions are compatible for the intended loss

    #     # Compute pairwise dot products for language with itself
    #     # aa should be (batch_size1,) then unsqueezed and expanded
    #     aa_dot = torch.einsum('bi,bi->b', anchor, anchor) # Result shape (batch_size1,)
        
    #     # If batch_size1 != batch_size2, this expansion might not be what's intended by original GRAM
    #     # For GRAM, typically all modalities in a "group" have the same batch dimension for comparison
    #     aa = aa_dot.unsqueeze(1).expand(-1, batch_size2) # Target shape (batch_size1, batch_size2)


    #     # Compute pairwise dot products for language with each input
    #     # l_inputs elements should be (batch_size1, batch_size2)
    #     l_inputs = []
    #     for input_tensor in inputs:
    #         # anchor: (batch_size1, dim), input_tensor: (batch_size2, dim)
    #         # input_tensor.T: (dim, batch_size2)
    #         # anchor @ input_tensor.T : (batch_size1, batch_size2)
    #         l_inputs.append(anchor @ input_tensor.T)


    #     # Compute pairwise dot products for each input with themselves and with each other
    #     # input_dot_products[i][j] should be (batch_size1, batch_size2)
    #     input_dot_products = []
    #     for i, input1 in enumerate(inputs): # input1 shape (batch_size2, dim)
    #         row = []
    #         for j, input2 in enumerate(inputs): # input2 shape (batch_size2, dim)
    #             # We need a result of shape (batch_size1, batch_size2)
    #             # The original einsum 'bi,bi->b' assumes input1 and input2 have same batch size (batch_size2)
    #             # and produces a result of (batch_size2,).
    #             # To make it (batch_size1, batch_size2), we unsqueeze and expand.
    #             # This implies that the dot product is computed for each item in input1/input2
    #             # and then this (scalar per pair) result is broadcasted across batch_size1.
    #             # This seems to be the intent of the original code's expand.
    #             dot_product_single_batch = torch.einsum('bi,bi->b', input1, input2) # Shape (batch_size2,)
    #             dot_product = dot_product_single_batch.unsqueeze(0).expand(batch_size1, -1) # Shape (batch_size1, batch_size2)
    #             row.append(dot_product)
    #         input_dot_products.append(row)

    #     # Stack the results to form the Gram matrix for each pair
    #     # G should be (batch_size1, batch_size2, num_modalities, num_modalities)
    #     # where num_modalities = 1 (anchor) + len(inputs)

    #     # First row of G_per_pair: [aa_pair, l_input1_pair, l_input2_pair, ...]
    #     # Subsequent rows: [l_input_i_pair, input_dot_products[i][0]_pair, input_dot_products[i][1]_pair, ...]
        
    #     # G will be constructed such that G[k,l,:,:] is the Gram matrix for the k-th anchor item and l-th input item set.
    #     # The dimensions of G before det should be (batch_size1, batch_size2, N, N) where N = 1 + len(inputs)
        
    #     # Let's construct G for each (anchor_item, input_set_item) pair
    #     # This means G will have dimensions (batch_size1, batch_size2, n_modalities, n_modalities)
        
    #     # Row for anchor
    #     anchor_row_elements = [aa] + l_inputs # Each element is (batch_size1, batch_size2)
        
    #     # Subsequent rows for inputs
    #     input_rows_elements = []
    #     for i in range(len(inputs)):
    #         # For the i-th input, the row is [l_inputs[i]] + input_dot_products[i]
    #         # l_inputs[i] is (batch_size1, batch_size2)
    #         # input_dot_products[i] is a list of tensors, each (batch_size1, batch_size2)
    #         current_input_row = [l_inputs[i]] + input_dot_products[i]
    #         #print(f"--- Debugging volume_computation stack for input_rows_elements, i={i} ---")
    #         #print(f"Shape of l_inputs[{i}]: {l_inputs[i].shape}")
    #         # for j_debug, tensor_in_row in enumerate(input_dot_products[i]):
    #         #     print(f"Shape of input_dot_products[{i}][{j_debug}]: {tensor_in_row.shape}")
    #         input_rows_elements.append(torch.stack(current_input_row, dim=-1)) # Stacks along new last dim -> (batch_size1, batch_size2, n_inputs_modalities)

    #     # Stack all rows
    #     # Anchor row needs to be stacked first
    #     stacked_anchor_row = torch.stack(anchor_row_elements, dim=-1) # (batch_size1, batch_size2, 1+n_inputs_modalities)
        
    #     all_rows_to_stack = [stacked_anchor_row] + input_rows_elements
        
    #     G = torch.stack(all_rows_to_stack, dim=-2) # Stack along the second to last dim
    #                                              # Resulting G shape: (batch_size1, batch_size2, n_modalities, n_modalities)
    #                                              # where n_modalities = 1 (anchor) + len(inputs)

    #     # Compute the determinant for each Gram matrix
    #     gram_det = torch.det(G.float()) # G.float() to ensure det works, result (batch_size1, batch_size2)

    #     # Compute the square root of the absolute value of the determinants
    #     res = torch.sqrt(torch.abs(gram_det))
    #     return res


    def volume_computation(self, language, video, audio):
        print(f"language shape: {language.shape}, video shape: {video.shape}, audio shape: {audio.shape}")
        A = torch.stack([language, video, audio], dim=1)
        A_T = A.transpose(-2, -1)
        G = A @ A_T
        gramian = torch.linalg.det(G.float())
        return torch.sqrt(gramian)


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and vocoder have their state dicts saved to CPU and then are moved to a `torch.device('meta')
        and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.vocoder]:
            cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLAP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        (
            bs_embed,
            seq_len,
        ) = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        if self.vocoder.config.model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {self.vocoder.config.model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents with width->self.vocoder.config.model_in_dim
    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            self.vocoder.config.model_in_dim // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the audio generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            audio_length_in_s (`int`, *optional*, defaults to 5.12):
                The length of the generated audio sample in seconds.
            num_inference_steps (`int`, *optional*, defaults to 10):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 2.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate audios that are closely linked to the text `prompt`,
                usually at the expense of lower sound quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate image. Choose between:
                - `"np"`: Return Numpy `np.ndarray` objects.
                - `"pt"`: Return PyTorch `torch.Tensor` objects.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated audios.
        """
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # print('latents: ', latents.shape) # [1, 8, 200, 16]

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                    class_labels=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # print('noise_pred.shape before: ', noise_pred.shape) # [2, 8, 200, 16]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        mel_spectrogram = self.decode_latents(latents)
        # print('mel: ', mel_spectrogram.shape)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)

        audio = audio[:, :original_waveform_length]

        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)
    


    def bind_prepare(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        learning_rate: float = 0.1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        eta: float = 0.0,
        video_paths: Union[str, List[str]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
    ):
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype


        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    
        out_dict = {'original_waveform_length': original_waveform_length,
                    'device': device,
                    'do_classifier_free_guidance': do_classifier_free_guidance,
                    'prompt_embeds': prompt_embeds,
                    'timesteps': timesteps,
                    # 'latents': latents,
                    'latents_dtype': latents_dtype,
                    'extra_step_kwargs': extra_step_kwargs,
                    }
        return out_dict, latents

    @torch.no_grad()
    def bind_step(
        self,
        original_waveform_length, 
        device,
        do_classifier_free_guidance,
        prompt_embeds,
        timesteps,
        latents,
        latents_dtype,
        extra_step_kwargs,
        # bind_model,
        # image_bind_video_input,
        cur_step,

        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        learning_rate: float = 0.1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        eta: float = 0.0,
        video_paths: Union[str, List[str]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # latents: Optional[torch.FloatTensor] = None,
        # prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # output_type: Optional[str] = "np",
    ):
        # image_bind_video_input = load_and_transform_video_data(video_paths, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps_bind = int(len(timesteps) * 0.2)

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        

        t = timesteps[cur_step]

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=None,
                class_labels=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample.to(dtype=latents_dtype)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample    
        
        # call the callback, if provided
        if cur_step == len(timesteps) - 1 or ((cur_step + 1) > num_warmup_steps and (cur_step + 1) % self.scheduler.order == 0):
            # progress_bar.update()
            if callback is not None and cur_step % callback_steps == 0:
                callback(cur_step, t, latents)

        return latents, noise_pred

    def xt2x0(
        self,
        latents_temp,
        timesteps,
        cur_step,
        noise_pred,
    ):
        t = timesteps[cur_step]

        x0 = 1/(self.scheduler.alphas_cumprod[t] ** 0.5) * (latents_temp - (1-self.scheduler.alphas_cumprod[t])**0.5 * noise_pred) 

        x0_mel_spectrogram = self.decode_latents(x0)

        if x0_mel_spectrogram.dim() == 4:
            x0_mel_spectrogram = x0_mel_spectrogram.squeeze(1)

        # 3. convert mel-spectrogram to waveform
        x0_waveform = self.vocoder(x0_mel_spectrogram) # TODO save this [1, 128032]

        return x0, x0_waveform

    def bind_finish(
        self,
        original_waveform_length, 
        latents,
        return_dict: bool = True,
        output_type: Optional[str] = "np",
    ):
        # 8. Post-processing
        mel_spectrogram = self.decode_latents(latents)
        # print('mel: ', mel_spectrogram.shape) # [1, 1, 800, 64]

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram) # [1, 128032]

        audio = audio[:, :original_waveform_length] # [1, 128000]

        if output_type == "np":
            # audio = audio.numpy()
            audio = audio.detach().numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)



    def bind_forward_double_loss(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        learning_rate: float = 0.1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        clip_duration: float = 2.0,
        clips_per_video: int = 5,
        num_optimization_steps: int = 1,
        optimization_starting_point: float = 0.2,
        eta: float = 0.0,
        video_paths: Union[str, List[str]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
        bind_model = None,
        gram_model = None,
        args_gram = None,
        use_gram_loss: bool = False,
        #gram_contrastive_temp: float = 0.07, # Parametro per GRAM loss
        #gram_label_smoothing: float = 0.1, # Parametro per GRAM loss
    ):
        if use_gram_loss and (gram_model is None or args_gram is None):
            raise ValueError("To use gram_loss, you must provide `gram_model` and `args_gram`.")
        # 0. Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        #image_bind_video_input = load_and_transform_video_data(video_paths, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)
        
        # for p in bind_model.parameters():
        #     p.requires_grad = False
    

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        num_warmup_steps_bind = int(len(timesteps) * optimization_starting_point)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        class_labels=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample    
                
                latents_temp = latents.detach()
                latents_temp.requires_grad = True

                optimizer = torch.optim.Adam([latents_temp], lr=learning_rate)
                
                if i > num_warmup_steps_bind:
                    for optim_step in range(num_optimization_steps):
                        with torch.autograd.set_detect_anomaly(True):
                            # 1. compute x0 
                            x0 = 1/(self.scheduler.alphas_cumprod[t] ** 0.5) * (latents_temp - (1-self.scheduler.alphas_cumprod[t])**0.5 * noise_pred) 
                            
                            # 2. decode x0 with vae decoder 
                            x0_mel_spectrogram = self.decode_latents(x0)

                            if x0_mel_spectrogram.dim() == 4:
                                x0_mel_spectrogram = x0_mel_spectrogram.squeeze(1)

                            # 3. convert mel-spectrogram to waveform
                            x0_waveform = self.vocoder(x0_mel_spectrogram)

                            calculated_loss = 0 # Inizializza la loss a 0

                            ######## USE GRAM LOSS ########
                            if use_gram_loss:
                                for p in gram_model.parameters():
                                    p.requires_grad = False

                                # Ora puoi chiamare build_batch direttamente con il tensore!
                                # Non c'è più bisogno di salvare su disco.
                                batch_for_gram = build_batch(
                                    args=args_gram,
                                    text=prompt,
                                    video=video_paths,
                                    audio=None,  # Ignorato perché forniamo la waveform
                                    audio_waveform=x0_waveform, # Passa il tensore
                                    device=device
                                )

                                evaluation_dict = gram_model(batch_for_gram, 'ret%tva', compute_loss=False)
                                feat_t = evaluation_dict['feat_t']
                                feat_v = evaluation_dict['feat_v']
                                feat_a = evaluation_dict['feat_a']

                                # Clear temporary audio files
                                for temp_path in temp_audio_files:
                                    os.remove(temp_path)

                                # Calculate the volume, we want to minimize it to maximize alignment.
                                volume = self.volume_computation(feat_t, feat_v, feat_a)
                                calculated_loss = volume.mean() # La loss è direttamente il volume.
                            

                            ######## USE BIND LOSS ########
                            else:
                                image_bind_video_input = load_and_transform_video_data(video_paths, device, clip_duration=clip_duration, clips_per_video=clips_per_video, n_samples_per_clip=2)
                                for p in bind_model.parameters():
                                    p.requires_grad = False
                
                                #print(f"x0_waveform shape: {x0_waveform.shape}")
                                # 4. waveform to imagebind mel-spectrogram
                                x0_imagebind_audio_input = load_and_transform_audio_data_from_waveform(x0_waveform, org_sample_rate=self.vocoder.config.sampling_rate, 
                                                    device=device, target_length=204, clip_duration=clip_duration, clips_per_video=clips_per_video)
                                #print(f"x0_imagebind_audio_input shape: {x0_imagebind_audio_input.shape}")
                                current_prompts_for_bind_model: List[str]
                                if isinstance(prompt, str):
                                    current_prompts_for_bind_model = [prompt] # Effective batch_size = 1
                                elif isinstance(prompt, list):
                                    current_prompts_for_bind_model = prompt # Effective batch_size = len(prompt)
                                else:
                                    # This case implies prompt_embeds were passed, and raw text is not available.
                                    # Text-audio loss cannot be computed directly.
                                    # You might want to raise an error or skip text loss.
                                    logger.warning("Raw text prompt not available for BIND text-loss calculation.")
                                    current_prompts_for_bind_model = None

                                inputs_for_bind = {
                                    ModalityType.AUDIO: x0_imagebind_audio_input,
                                }

                                if current_prompts_for_bind_model is not None:
                                    inputs_for_bind[ModalityType.TEXT] = load_and_transform_text(current_prompts_for_bind_model, device)
                                
                                if video_paths is not None: # image_bind_video_input was loaded outside the loop
                                    inputs_for_bind[ModalityType.VISION] = image_bind_video_input.detach() if isinstance(image_bind_video_input, torch.Tensor) else image_bind_video_input
                                
                                # Cast inputs to float16 if necessary
                                for k_bind in inputs_for_bind:
                                    if isinstance(inputs_for_bind[k_bind], torch.Tensor):
                                        if inputs_for_bind[k_bind].dtype in [torch.float32, torch.float64]:
                                            inputs_for_bind[k_bind] = inputs_for_bind[k_bind].to(dtype=torch.float32)
                                    elif isinstance(inputs_for_bind[k_bind], list): # Should not happen with current load_and_transform
                                        inputs_for_bind[k_bind] = [
                                            x.to(dtype=torch.float32) if (isinstance(x, torch.Tensor) and x.dtype in [torch.float32, torch.float64]) else x
                                            for x in inputs_for_bind[k_bind]
                                        ]
                            
                                embeddings = bind_model(inputs_for_bind)
                                #normalize embeddings
                                audio_embeds = F.normalize(embeddings[ModalityType.AUDIO], dim=1, p=2) # Shape: (batch_size * num_waveforms_per_prompt, embed_dim)
                                #print(f"Shape of audio_embeds: {audio_embeds.shape}")
                                
                                accumulated_cosine_loss_per_sample = torch.zeros_like(audio_embeds[:, 0], device=audio_embeds.device, dtype=audio_embeds.dtype)
                                loss_components_cosine = 0
                                
                                if ModalityType.TEXT in embeddings:
                                    text_embeds = embeddings[ModalityType.TEXT] # Shape: (batch_size, embed_dim)
                                    text_embeds_repeated = text_embeds.repeat_interleave(num_waveforms_per_prompt, dim=0)
                                    bind_loss_text_audio_per_sample = 1 - F.cosine_similarity(text_embeds_repeated, audio_embeds) # Tensore (N,)
                                    accumulated_cosine_loss_per_sample += bind_loss_text_audio_per_sample
                                    loss_components_cosine +=1

                                if ModalityType.VISION in embeddings:
                                    vision_embeds = embeddings[ModalityType.VISION] # Shape: (batch_size, embed_dim)
                                    vision_embeds_repeated = vision_embeds.repeat_interleave(num_waveforms_per_prompt, dim=0)
                                    bind_loss_vision_audio_per_sample = 1 - F.cosine_similarity(vision_embeds_repeated, audio_embeds) # Tensore (N,)
                                    accumulated_cosine_loss_per_sample += bind_loss_vision_audio_per_sample
                                    loss_components_cosine +=1
                                
                                if loss_components_cosine > 0:
                                    calculated_loss = accumulated_cosine_loss_per_sample.mean()

                            if calculated_loss != 0: # Check if any loss was actually calculated
                                print(f"Calculated loss: {calculated_loss.item()}")
                                calculated_loss.backward() 
                                optimizer.step()
                            optimizer.zero_grad()

                latents = latents_temp.detach()


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        mel_spectrogram = self.decode_latents(latents)

        audio = self.mel_spectrogram_to_waveform(mel_spectrogram) # [1, 128032]
        audio = audio[:, :original_waveform_length] # [1, 128000]

        if output_type == "np":
            # audio = audio.numpy()
            audio = audio.detach().numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)


    def only_prepare_latents(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        guidance_scale: float = 2.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "np",
    ):
        vocoder_upsample_factor = np.prod(self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.vocoder.config.sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_length_in_s,
            vocoder_upsample_factor,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        return latents
    