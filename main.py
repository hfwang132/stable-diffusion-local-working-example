import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import gradio as gr
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Model options with their corresponding Hugging Face model IDs
models = {
    "Anything v3.0 (Best)": "Linaqruf/anything-v3.0",
    "Anime Kawaii Diffusion (Good)": "Ojimi/anime-kawai-diffusion",
    "Waifu Diffusion (Not very good)": "hakurei/waifu-diffusion",
}

# Function to load the Real-ESRGAN model
def load_esrgan_model(model_path, scale_factor):
    # Define the model architecture (RRDBNet)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_factor)
    
    # Load the Real-ESRGANer utility with the given model and weights
    upsampler = RealESRGANer(
        scale=scale_factor,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True  # Use half-precision
    )
    return upsampler

# Function to upscale an image using Real-ESRGAN
def upscale_image(img, model):
    # Convert the image to a NumPy array (expected by the Real-ESRGAN model)
    img_np = np.array(img)

    # Perform the upscaling
    output_img_np, _ = model.enhance(img_np)

    # Convert the output NumPy array back to a PIL image
    output_img = Image.fromarray(np.uint8(output_img_np))
    
    return output_img

# Function to generate images
def generate_images(model_name, prompt, negative_prompt, num_steps, guidance_scale, height, width, batch_size, upscale, resize_factor):
    # Retrieve the actual model ID from the dictionary
    model_id = models[model_name]
    
    # Load the selected model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    pipe.to("cuda")
    pipe.safety_checker = None

    # Generate images with the given prompt and negative prompt
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=batch_size,
    ).images

    # If upscale is enabled, use Real-ESRGAN for upscaling
    if upscale:
        # Load the Real-ESRGAN model
        model_path = 'weights/RealESRGAN_x4plus.pth'  # Adjust this path as needed
        esrgan_model = load_esrgan_model(model_path, resize_factor)

        # Upscale each image using Real-ESRGAN
        upscaled_images = []
        for img in images:
            img_upscaled = upscale_image(img, esrgan_model)
            upscaled_images.append(img_upscaled)

        return upscaled_images

    return images

# Gradio interface with options to optimize performance
def gradio_interface():
    with gr.Blocks() as demo:
        # Input fields for user parameters
        with gr.Row():
            model_selection = gr.Dropdown(label="Select Model", choices=list(models.keys()), value="Anything v3.0 (Best)")
            prompt_text = """1 girl, masterpiece, best quality, absurdres, newest, ruby eyes, fox ears, brown hair, smile, medium boobs, loli, half naked, mini bikini, cute, kawaii, stockings, looking at viewer, sitting on a bed, long black gloves
            """
            negative_prompt_text = """(hands:1.5), (fingers), bad anatomy, extra limbs, extra feet, bad proportions, extra crus, fused crus, three legs, amputation, (vagina:1.5), blurred, texts, extra belly button
            """
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the anime scene...", 
                                value=prompt_text)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Avoid certain elements (e.g., blurry, dark)", 
                                         value=negative_prompt_text)
            num_steps = gr.Slider(1, 100, value=20, step=1, label="Number of Diffusion Steps")
            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
            height = gr.Slider(256, 1024, value=512, step=8, label="Image Height")
            width = gr.Slider(256, 1024, value=512, step=8, label="Image Width")
            batch_size = gr.Slider(1, 20, value=2, step=1, label="Number of Images to Generate")
            upscale = gr.Checkbox(label="Upscale Images", value=False)
            resize_factor = gr.Slider(1.0, 4.0, value=4.0, step=0.1, label="Resize Factor (e.g., 2x for upscaling)")

        # Image display
        with gr.Row():
            output = gr.Gallery(label="Generated Images", height="auto")

        # Action button
        with gr.Row():
            generate_button = gr.Button("Generate Images")

        # Link the action button to the generation function
        generate_button.click(generate_images, 
                              inputs=[model_selection, prompt, negative_prompt, num_steps, guidance_scale, height, width, batch_size, upscale, resize_factor], 
                              outputs=output)

    return demo

# Launch the interface
gradio_interface().launch()
