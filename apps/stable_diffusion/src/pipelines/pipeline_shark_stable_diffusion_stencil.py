import torch
import time
import numpy as np
from tqdm.auto import tqdm
from random import randint
from PIL import Image
from transformers import CLIPTokenizer
from typing import Union
from shark.shark_inference import SharkInference
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DDPMScheduler,
    KDPM2DiscreteScheduler,
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.utils import controlnet_hint_conversion
from apps.stable_diffusion.src.utils import (
    start_profiling,
    end_profiling,
)
from apps.stable_diffusion.src.models import SharkifyStableDiffusionModel


class StencilPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
            DEISMultistepScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
            DDPMScheduler,
            KDPM2DiscreteScheduler,
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        ondemand: bool,
        controlnet_names: list[str],
    ):
        super().__init__(scheduler, sd_model, import_mlir, use_lora, ondemand)
        self.controlnet = [None] * len(controlnet_names)
        self.controlnet_512 = [None] * len(controlnet_names)
        self.controlnet_id = [str] * len(controlnet_names)
        self.controlnet_512_id = [str] * len(controlnet_names)
        self.controlnet_names = controlnet_names

    def load_controlnet(self, index, model_name):
        if model_name is None:
            return
        if (
            self.controlnet[index] is not None
            and self.controlnet_id[index] is not None
            and self.controlnet_id[index] == model_name
        ):
            return
        self.controlnet_id[index] = model_name
        self.controlnet[index] = self.sd_model.controlnet(model_name)

    def unload_controlnet(self, index):
        del self.controlnet[index]
        self.controlnet_id[index] = None
        self.controlnet[index] = None

    def load_controlnet_512(self, index, model_name):
        if (
            self.controlnet_512[index] is not None
            and self.controlnet_512_id[index] == model_name
        ):
            return
        self.controlnet_512_id[index] = model_name
        self.controlnet_512[index] = self.sd_model.controlnet(
            model_name, use_large=True
        )

    def unload_controlnet_512(self, index):
        del self.controlnet_512[index]
        self.controlnet_512_id[index] = None
        self.controlnet_512[index] = None

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        generator,
        num_inference_steps,
        dtype,
    ):
        latents = torch.randn(
            (
                batch_size,
                4,
                height // 8,
                width // 8,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def produce_stencil_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
        stencil_hints=[None],
        controlnet_conditioning_scale: float = 1.0,
        mask=None,
        masked_image_latents=None,
        return_all_latents=False,
    ):
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
        if text_embeddings.shape[1] <= self.model_max_length:
            self.load_unet()
        else:
            self.load_unet_512()

        for i, name in enumerate(self.controlnet_names):
            if text_embeddings.shape[1] <= self.model_max_length:
                self.load_controlnet(i, name)
            else:
                self.load_controlnet_512(i, name)

        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            timestep = torch.tensor([t]).to(dtype)
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            if mask is not None and masked_image_latents is not None:
                latent_model_input = torch.cat(
                    [
                        torch.from_numpy(np.asarray(latent_model_input)),
                        mask,
                        masked_image_latents,
                    ],
                    dim=1,
                ).to(dtype)
            if cpu_scheduling:
                latent_model_input = latent_model_input.detach().numpy()

            if not torch.is_tensor(latent_model_input):
                latent_model_input_1 = torch.from_numpy(
                    np.asarray(latent_model_input)
                ).to(dtype)
            else:
                latent_model_input_1 = latent_model_input

            # Multicontrolnet
            control = None
            for i, controlnet_hint in enumerate(stencil_hints):
                if controlnet_hint is None:
                    continue
                if text_embeddings.shape[1] <= self.model_max_length:
                    subcontrol = self.controlnet[i](
                        "forward",
                        (
                            latent_model_input_1,
                            timestep,
                            text_embeddings,
                            controlnet_hint,
                        ),
                        send_to_host=False,
                    )
                else:
                    subcontrol = self.controlnet_512[i](
                        "forward",
                        (
                            latent_model_input_1,
                            timestep,
                            text_embeddings,
                            controlnet_hint,
                        ),
                        send_to_host=False,
                    )
                subcontrol = [
                    torch.from_numpy(np.asarray(x)) for x in subcontrol
                ]
                # sizes = [x.shape for x in subcontrol]
                # subcontrol = [torch.mean(x, dim=(2, 3), keepdim=True) for x in subcontrol]
                # subcontrol = [x.broadcast_to(y) for (x, y) in zip(subcontrol, sizes)]
                if control is None:
                    control = subcontrol
                else:
                    control = [x + y for (x, y) in zip(control, subcontrol)]

            timestep = timestep.detach().numpy()
            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            # TODO: Pass `control` as it is to Unet. Same as TODO mentioned in model_wrappers.py.

            if text_embeddings.shape[1] <= self.model_max_length:
                noise_pred = self.unet(
                    "forward",
                    (
                        latent_model_input,
                        timestep,
                        text_embeddings_numpy,
                        guidance_scale,
                        control[0],
                        control[1],
                        control[2],
                        control[3],
                        control[4],
                        control[5],
                        control[6],
                        control[7],
                        control[8],
                        control[9],
                        control[10],
                        control[11],
                        control[12],
                    ),
                    send_to_host=False,
                )
            else:
                print(self.unet_512)
                noise_pred = self.unet_512(
                    "forward",
                    (
                        latent_model_input,
                        timestep,
                        text_embeddings_numpy,
                        guidance_scale,
                        control[0],
                        control[1],
                        control[2],
                        control[3],
                        control[4],
                        control[5],
                        control[6],
                        control[7],
                        control[8],
                        control[9],
                        control[10],
                        control[11],
                        control[12],
                    ),
                    send_to_host=False,
                )
            end_profiling(profile_device)

            if cpu_scheduling:
                noise_pred = torch.from_numpy(noise_pred.to_host())
                latents = self.scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents)

            latent_history.append(latents)
            step_time = (time.time() - step_start_time) * 1000
            #  self.log += (
            #      f"\nstep = {i} | timestep = {t} | time = {step_time:.2f}ms"
            #  )
            step_time_sum += step_time

        if self.ondemand:
            self.unload_unet()
            self.unload_unet_512()
            for i in range(len(self.controlnet_names)):
                self.unload_controlnet(i)
                self.unload_controlnet_512(i)
        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        batch_size,
        height,
        width,
        num_inference_steps,
        strength,
        guidance_scale,
        seed,
        max_length,
        dtype,
        use_base_vae,
        cpu_scheduling,
        max_embeddings_multiples,
        stencils,
        stencil_images,
        resample_type,
    ):
        # Control Embedding check & conversion
        # TODO: 1. Change `num_images_per_prompt`.
        # controlnet_hint = controlnet_hint_conversion(
        #     image, use_stencil, height, width, dtype, num_images_per_prompt=1
        # )
        stencil_hints = []
        for i, stencil in enumerate(stencils):
            image = stencil_images[i]
            stencil_hints.append(
                controlnet_hint_conversion(
                    image,
                    stencil,
                    height,
                    width,
                    dtype,
                    num_images_per_prompt=1,
                )
            )

        # prompts and negative prompts must be a list.
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts]

        prompts = prompts * batch_size
        neg_prompts = neg_prompts * batch_size

        # seed generator to create the inital latent noise. Also handle out of range seeds.
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(seed)

        # Get text embeddings with weight emphasis from prompts
        text_embeddings = self.encode_prompts_weight(
            prompts,
            neg_prompts,
            max_length,
            max_embeddings_multiples=max_embeddings_multiples,
        )

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Prepare initial latent.
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )
        final_timesteps = self.scheduler.timesteps

        # Get Image latents
        latents = self.produce_stencil_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            stencil_hints=stencil_hints,
        )

        # Img latents -> PIL images
        all_imgs = []
        self.load_vae()
        for i in tqdm(range(0, latents.shape[0], batch_size)):
            imgs = self.decode_latents(
                latents=latents[i : i + batch_size],
                use_base_vae=use_base_vae,
                cpu_scheduling=cpu_scheduling,
            )
            all_imgs.extend(imgs)
        if self.ondemand:
            self.unload_vae()

        return all_imgs
