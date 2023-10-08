import os
import torch
import time
import gradio as gr
import PIL
from math import ceil
from PIL import Image
import base64
from io import BytesIO
from fastapi.exceptions import HTTPException
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list_cpu_only,
    predefined_models,
    cancel_sd,
)
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
    StencilPipeline,
    resize_stencil,
    get_schedulers,
    set_init_device_flags,
    utils,
    save_output_img,
)
from apps.stable_diffusion.src.utils import (
    get_generated_imgs_path,
    get_generation_text_info,
)
from apps.stable_diffusion.src.utils.stencils import (
    CannyDetector,
    OpenposeDetector,
)
from apps.stable_diffusion.web.utils.common_label_calc import status_label
import numpy as np


# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


# Stencils
# TODO: Add more stencils here
STENCIL_COUNT = 2
stencils = [None] * STENCIL_COUNT
images = [None] * STENCIL_COUNT


# Exposed to UI.
def img2img_inf(
    prompt: str,
    negative_prompt: str,
    image_dict,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: str | int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    custom_model: str,
    hf_model_id: str,
    custom_vae: str,
    precision: str,
    device: str,
    max_length: int,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    lora_weights: str,
    lora_hf_id: str,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
):
    from apps.stable_diffusion.web.ui.utils import (
        get_custom_model_pathfile,
        get_custom_vae_or_lora_weights,
        Config,
    )
    import apps.stable_diffusion.web.utils.global_obj as global_obj
    from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
        SD_STATE_CANCEL,
    )

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.seed = seed
    args.steps = steps
    args.strength = strength
    args.scheduler = scheduler
    args.img_path = "not none"
    args.ondemand = ondemand

    global stencils
    global images

    for i, stencil in enumerate(stencils):
        if images[i] is None and stencil is not None:
            return None, "A stencil must have an Image input"
        if images[i] is not None:
            images[i] = images[i].convert("RGB")

    if image_dict is None:
        return None, "An Initial Image is required"
    # if use_stencil == "scribble":
    #     image = image_dict["mask"].convert("RGB")
    if isinstance(image_dict, PIL.Image.Image):
        image = image_dict.convert("RGB")
    else:
        image = image_dict["image"].convert("RGB")

    # set ckpt_loc and hf_model_id.
    args.ckpt_loc = ""
    args.hf_model_id = ""
    args.custom_vae = ""
    if custom_model == "None":
        if not hf_model_id:
            return (
                None,
                "Please provide either custom model or huggingface model ID, "
                "both must not be empty.",
            )
        if "civitai" in hf_model_id:
            args.ckpt_loc = hf_model_id
        else:
            args.hf_model_id = hf_model_id
    elif ".ckpt" in custom_model or ".safetensors" in custom_model:
        args.ckpt_loc = get_custom_model_pathfile(custom_model)
    else:
        args.hf_model_id = custom_model
    if custom_vae != "None":
        args.custom_vae = get_custom_model_pathfile(custom_vae, model="vae")

    args.use_lora = get_custom_vae_or_lora_weights(
        lora_weights, lora_hf_id, "lora"
    )

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    stencil_count = 0
    for stencil in stencils:
        if stencil is not None:
            stencil_count += 1
    if stencil_count > 0:
        args.scheduler = "DDIM"
        args.hf_model_id = "runwayml/stable-diffusion-v1-5"
        # image, width, height = resize_stencil(image)
    elif "Shark" in args.scheduler:
        print(
            f"Shark schedulers are not supported. Switching to EulerDiscrete "
            f"scheduler"
        )
        args.scheduler = "EulerDiscrete"
    cpu_scheduling = not args.scheduler.startswith("Shark")
    args.precision = precision
    dtype = torch.float32 if precision == "fp32" else torch.half
    new_config_obj = Config(
        "img2img",
        args.hf_model_id,
        args.ckpt_loc,
        args.custom_vae,
        precision,
        batch_size,
        max_length,
        height,
        width,
        device,
        use_lora=args.use_lora,
        stencils=stencils,
        ondemand=ondemand,
    )
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_cfg_obj() != new_config_obj
    ):
        global_obj.clear_cache()
        global_obj.set_cfg_obj(new_config_obj)
        args.batch_count = batch_count
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.iree_vulkan_target_triple = init_iree_vulkan_target_triple
        args.use_tuned = init_use_tuned
        args.import_mlir = init_import_mlir
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-1-base"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(args.scheduler)

        if stencil_count > 0:
            args.use_tuned = False
            global_obj.set_sd_obj(
                StencilPipeline.from_pretrained(
                    scheduler_obj,
                    args.import_mlir,
                    args.hf_model_id,
                    args.ckpt_loc,
                    args.custom_vae,
                    args.precision,
                    args.max_length,
                    args.batch_size,
                    args.height,
                    args.width,
                    args.use_base_vae,
                    args.use_tuned,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    stencils=stencils,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                    ondemand=args.ondemand,
                )
            )
        else:
            global_obj.set_sd_obj(
                Image2ImagePipeline.from_pretrained(
                    scheduler_obj,
                    args.import_mlir,
                    args.hf_model_id,
                    args.ckpt_loc,
                    args.custom_vae,
                    args.precision,
                    args.max_length,
                    args.batch_size,
                    args.height,
                    args.width,
                    args.use_base_vae,
                    args.use_tuned,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                    ondemand=args.ondemand,
                )
            )

    global_obj.set_sd_scheduler(args.scheduler)

    start_time = time.time()
    global_obj.get_sd_obj().log = ""
    generated_imgs = []
    extra_info = {"STRENGTH": strength}
    text_output = ""
    try:
        seeds = utils.batch_seeds(seed, batch_count, repeatable_seeds)
    except TypeError as error:
        raise gr.Error(str(error)) from None

    for current_batch in range(batch_count):
        out_imgs = global_obj.get_sd_obj().generate_images(
            prompt,
            negative_prompt,
            image,
            batch_size,
            height,
            width,
            ceil(steps / strength),
            strength,
            guidance_scale,
            seeds[current_batch],
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            args.max_embeddings_multiples,
            stencils,
            images,
            resample_type=resample_type,
        )
        total_time = time.time() - start_time
        text_output = get_generation_text_info(
            seeds[: current_batch + 1], device
        )
        text_output += "\n" + global_obj.get_sd_obj().log
        text_output += f"\nTotal image(s) generation time: {total_time:.4f}sec"

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(
                out_imgs[0],
                seeds[current_batch],
                extra_info,
            )
            generated_imgs.extend(out_imgs)
            yield generated_imgs, text_output, status_label(
                "Image-to-Image", current_batch + 1, batch_count, batch_size
            )

    return generated_imgs, text_output, ""


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";", 1)[1].split(",", 1)[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail="Invalid encoded image")


def encode_pil_to_base64(images):
    encoded_imgs = []
    for image in images:
        with BytesIO() as output_bytes:
            if args.output_img_format.lower() == "png":
                image.save(output_bytes, format="PNG")

            elif args.output_img_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG")
            else:
                raise HTTPException(
                    status_code=500, detail="Invalid image format"
                )
            bytes_data = output_bytes.getvalue()
            encoded_imgs.append(base64.b64encode(bytes_data))
    return encoded_imgs


# Img2Img Rest API.
def img2img_api(
    InputData: dict,
):
    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"]}.'
    )
    init_image = decode_base64_to_image(InputData["init_images"][0])
    res = img2img_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        init_image,
        InputData["height"],
        InputData["width"],
        InputData["steps"],
        InputData["denoising_strength"],
        InputData["cfg_scale"],
        InputData["seed"],
        batch_count=1,
        batch_size=1,
        scheduler="EulerDiscrete",
        custom_model="None",
        hf_model_id=InputData["hf_model_id"]
        if "hf_model_id" in InputData.keys()
        else "stabilityai/stable-diffusion-2-1-base",
        custom_vae="None",
        precision="fp16",
        device=available_devices[0],
        max_length=64,
        use_stencil=InputData["use_stencil"]
        if "use_stencil" in InputData.keys()
        else "None",
        save_metadata_to_json=False,
        save_metadata_to_png=False,
        lora_weights="None",
        lora_hf_id="",
        ondemand=False,
        repeatable_seeds=False,
        resample_type="Lanczos",
    )

    # Converts generator type to subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


with gr.Blocks(title="Image-to-Image") as img2img_web:
    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Row():
            with gr.Column(scale=1, elem_id="demo_title_outer"):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                    width=150,
                    height=50,
                )
    with gr.Row(elem_id="ui_body"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Row():
                    # janky fix for overflowing text
                    i2i_model_info = (str(get_custom_model_path())).replace(
                        "\\", "\n\\"
                    )
                    i2i_model_info = f"Custom Model Path: {i2i_model_info}"
                    img2img_custom_model = gr.Dropdown(
                        label=f"Models",
                        info=i2i_model_info,
                        elem_id="custom_model",
                        value=os.path.basename(args.ckpt_loc)
                        if args.ckpt_loc
                        else "stabilityai/stable-diffusion-2-1-base",
                        choices=["None"]
                        + get_custom_model_files()
                        + predefined_models,
                    )
                    img2img_hf_model_id = gr.Textbox(
                        elem_id="hf_model_id",
                        placeholder="Select 'None' in the Models dropdown "
                        "on the left and enter model ID here "
                        "e.g: SG161222/Realistic_Vision_V1.3, "
                        "https://civitai.com/api/download/models/15236",
                        value="",
                        label="HuggingFace Model ID or Civitai model "
                        "download URL",
                        lines=3,
                    )
                    # janky fix for overflowing text
                    i2i_vae_info = (str(get_custom_model_path("vae"))).replace(
                        "\\", "\n\\"
                    )
                    i2i_vae_info = f"VAE Path: {i2i_vae_info}"
                    custom_vae = gr.Dropdown(
                        label=f"Custom VAE Models",
                        info=i2i_vae_info,
                        elem_id="custom_model",
                        value=os.path.basename(args.custom_vae)
                        if args.custom_vae
                        else "None",
                        choices=["None"] + get_custom_model_files("vae"),
                    )

                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=args.prompts[0],
                        lines=2,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=args.negative_prompts[0],
                        lines=2,
                        elem_id="negative_prompt_box",
                    )
                # TODO: make this import image prompt info if it exists
                img2img_init_image = gr.Image(
                    label="Input Image",
                    source="upload",
                    tool="sketch",
                    type="pil",
                    height=300,
                )

                # with gr.Accordion(label="Stencil Options", open=False):
                #     with gr.Row():
                #         use_stencil = gr.Dropdown(
                #             elem_id="stencil_model",
                #             label="Stencil model",
                #             value="None",
                #             choices=["None", "canny", "openpose", "scribble"],
                #         )

                #     def show_canvas(choice):
                #         if choice == "scribble":
                #             return (
                #                 gr.Slider.update(visible=True),
                #                 gr.Slider.update(visible=True),
                #                 gr.Button.update(visible=True),
                #             )
                #         else:
                #             return (
                #                 gr.Slider.update(visible=False),
                #                 gr.Slider.update(visible=False),
                #                 gr.Button.update(visible=False),
                #             )

                #     def create_canvas(w, h):
                #         return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

                #     with gr.Row():
                #         canvas_width = gr.Slider(
                #             label="Canvas Width",
                #             minimum=256,
                #             maximum=1024,
                #             value=512,
                #             step=1,
                #             visible=False,
                #         )
                #         canvas_height = gr.Slider(
                #             label="Canvas Height",
                #             minimum=256,
                #             maximum=1024,
                #             value=512,
                #             step=1,
                #             visible=False,
                #         )
                #     create_button = gr.Button(
                #         label="Start",
                #         value="Open drawing canvas!",
                #         visible=False,
                #     )
                #     create_button.click(
                #         fn=create_canvas,
                #         inputs=[canvas_width, canvas_height],
                #         outputs=[img2img_init_image],
                #     )
                #     use_stencil.change(
                #         fn=show_canvas,
                #         inputs=use_stencil,
                #         outputs=[canvas_width, canvas_height, create_button],
                #     )

                with gr.Accordion(label="Multistencil Options", open=False):
                    choices = ["None", "canny", "openpose", "scribble"]

                    def cnet_preview(checked, model, input_image, index):
                        global stencils
                        global images
                        if not checked:
                            stencils[index] = None
                            images[index] = None
                            return None
                        images[index] = input_image
                        stencils[index] = model
                        match model:
                            case "canny":
                                canny = CannyDetector()
                                result = canny(np.array(input_image), 100, 200)
                                return [Image.fromarray(result), result]
                            case "openpose":
                                openpose = OpenposeDetector()
                                result = openpose(np.array(input_image))
                                # TODO: This is just an empty canvas, need to draw the candidates (which are in result[1])
                                return [Image.fromarray(result[0]), result]
                            case _:
                                return None

                    with gr.Row():
                        cnet_1 = gr.Checkbox(show_label=False)
                        cnet_1_model = gr.Dropdown(
                            label="Controlnet 1",
                            value="None",
                            choices=choices,
                        )
                        cnet_1_image = gr.Image(
                            source="upload",
                            tool=None,
                            type="pil",
                        )
                        cnet_1_output = gr.Gallery(
                            show_label=False,
                            object_fit="scale-down",
                            rows=1,
                            columns=1,
                        )
                        cnet_1.change(
                            fn=(lambda a, b, c: cnet_preview(a, b, c, 0)),
                            inputs=[cnet_1, cnet_1_model, cnet_1_image],
                            outputs=[cnet_1_output],
                        )
                    with gr.Row():
                        cnet_2 = gr.Checkbox(show_label=False)
                        cnet_2_model = gr.Dropdown(
                            label="Controlnet 2",
                            value="None",
                            choices=choices,
                        )
                        cnet_2_image = gr.Image(
                            source="upload",
                            tool=None,
                            type="pil",
                        )
                        cnet_2_output = gr.Gallery(
                            show_label=False,
                            object_fit="scale-down",
                            rows=1,
                            columns=1,
                        )
                        cnet_2.change(
                            fn=(lambda a, b, c: cnet_preview(a, b, c, 1)),
                            inputs=[cnet_2, cnet_2_model, cnet_2_image],
                            outputs=[cnet_2_output],
                        )

                with gr.Accordion(label="LoRA Options", open=False):
                    with gr.Row():
                        # janky fix for overflowing text
                        i2i_lora_info = (
                            str(get_custom_model_path("lora"))
                        ).replace("\\", "\n\\")
                        i2i_lora_info = f"LoRA Path: {i2i_lora_info}"
                        lora_weights = gr.Dropdown(
                            label=f"Standalone LoRA Weights",
                            info=i2i_lora_info,
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_custom_model_files("lora"),
                        )
                        lora_hf_id = gr.Textbox(
                            elem_id="lora_hf_id",
                            placeholder="Select 'None' in the Standalone LoRA "
                            "weights dropdown on the left if you want to use "
                            "a standalone HuggingFace model ID for LoRA here "
                            "e.g: sayakpaul/sd-model-finetuned-lora-t4",
                            value="",
                            label="HuggingFace Model ID",
                            lines=3,
                        )
                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="EulerDiscrete",
                            choices=scheduler_list_cpu_only,
                        )
                        with gr.Group():
                            save_metadata_to_png = gr.Checkbox(
                                label="Save prompt information to PNG",
                                value=args.write_metadata_to_png,
                                interactive=True,
                            )
                            save_metadata_to_json = gr.Checkbox(
                                label="Save prompt information to JSON file",
                                value=args.save_metadata_to_json,
                                interactive=True,
                            )
                    with gr.Row():
                        height = gr.Slider(
                            384, 768, value=args.height, step=8, label="Height"
                        )
                        width = gr.Slider(
                            384, 768, value=args.width, step=8, label="Width"
                        )
                        max_length = gr.Radio(
                            label="Max Length",
                            value=args.max_length,
                            choices=[
                                64,
                                77,
                            ],
                            visible=False,
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            steps = gr.Slider(
                                1, 100, value=args.steps, step=1, label="Steps"
                            )
                        with gr.Column(scale=3):
                            strength = gr.Slider(
                                0,
                                1,
                                value=args.strength,
                                step=0.01,
                                label="Denoising Strength",
                            )
                            resample_type = gr.Dropdown(
                                value=args.resample_type,
                                choices=[
                                    "Lanczos",
                                    "Nearest Neighbor",
                                    "Bilinear",
                                    "Bicubic",
                                    "Adaptive",
                                    "Antialias",
                                    "Box",
                                    "Affine",
                                    "Cubic",
                                ],
                                label="Resample Type",
                            )
                        ondemand = gr.Checkbox(
                            value=args.ondemand,
                            label="Low VRAM",
                            interactive=True,
                        )
                        precision = gr.Radio(
                            label="Precision",
                            value=args.precision,
                            choices=[
                                "fp16",
                                "fp32",
                            ],
                            visible=True,
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=args.guidance_scale,
                                step=0.1,
                                label="CFG Scale",
                            )
                        with gr.Column(scale=3):
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=args.batch_count,
                                step=1,
                                label="Batch Count",
                                interactive=True,
                            )
                        repeatable_seeds = gr.Checkbox(
                            args.repeatable_seeds,
                            label="Repeatable Seeds",
                        )
                    with gr.Row():
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=args.batch_size,
                            step=1,
                            label="Batch Size",
                            interactive=False,
                            visible=False,
                        )
                with gr.Row():
                    seed = gr.Textbox(
                        value=args.seed,
                        label="Seed",
                        info="An integer or a JSON list of integers, -1 for random",
                    )
                    device = gr.Dropdown(
                        elem_id="device",
                        label="Device",
                        value=available_devices[0],
                        choices=available_devices,
                    )
                with gr.Row():
                    random_seed = gr.Button("Randomize Seed")
                    random_seed.click(
                        lambda: -1,
                        inputs=[],
                        outputs=[seed],
                        queue=False,
                    )
                    stop_batch = gr.Button("Stop Batch")
                    stable_diffusion = gr.Button("Generate Image(s)")

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    img2img_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=1,
                        elem_id="std_output",
                        show_label=False,
                    )
                    img2img_status = gr.Textbox(visible=False)
                with gr.Row():
                    img2img_sendto_inpaint = gr.Button(value="SendTo Inpaint")
                    img2img_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )
                    img2img_sendto_upscaler = gr.Button(
                        value="SendTo Upscaler"
                    )

        kwargs = dict(
            fn=img2img_inf,
            inputs=[
                prompt,
                negative_prompt,
                img2img_init_image,
                height,
                width,
                steps,
                strength,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                img2img_custom_model,
                img2img_hf_model_id,
                custom_vae,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                lora_weights,
                lora_hf_id,
                ondemand,
                repeatable_seeds,
                resample_type,
            ],
            outputs=[img2img_gallery, std_output, img2img_status],
            show_progress="minimal" if args.progress_bar else "none",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Image-to-Image", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=img2img_status,
        )

        prompt_submit = prompt.submit(**status_kwargs).then(**kwargs)
        neg_prompt_submit = negative_prompt.submit(**status_kwargs).then(
            **kwargs
        )
        generate_click = stable_diffusion.click(**status_kwargs).then(**kwargs)
        stop_batch.click(
            fn=cancel_sd,
            cancels=[prompt_submit, neg_prompt_submit, generate_click],
        )
