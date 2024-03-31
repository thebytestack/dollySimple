import random
import gradio as gr
from PIL import Image
import torch
import uuid
import numpy as np
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# we have give lora weights here
adapter_id = "qwe_cat_long.safetensors"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, 
                                                        torch_dtype=torch.float16, 
                                                        variant="fp16")
pipe.to("cuda")

# load lora

pipe.load_lora_weights('lora_weights',
                       weight_name='qwe_cat_long.safetensors', 
                       )
pipe.fuse_lora()

def generate(text, comp_dict, guidance_scale, num_images_per_prompt):
    print(comp_dict.keys())
    image = Image.fromarray(comp_dict['background'])
    mask = Image.fromarray(comp_dict['layers'][0][:,:,0])

    prompt = text
    image1 = pipe(prompt=prompt, 
                 negative_prompt='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting',
                 guidance_scale=guidance_scale,
                 mask_image=mask,
                 image=image,
                 num_images_per_prompt=num_images_per_prompt,
                 height=576,
                 width=1024,
                 num_inference_steps=30).images
    
    print(image1)
    
    # a= np.array(image)
    # b = np.array(image1)
    # a_resized = np.resize(a, b.shape)
    # stacked = np.hstack((a_resized, b))
    # img = Image.fromarray(stacked)
    # img.save(f'{uuid.uuid1()}.png')

    return image1



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():

            input_image = gr.ImageEditor(image_mode='RGB')
            text = gr.Textbox(
                    label="Enter Prompt...")
            
            btn = gr.Button("Generate", scale=0)
            
            guidance_scale = gr.Slider(minimum=0, maximum=15, value=7.5, label='guidance scale')
            num_images_per_prompt = gr.Slider(minimum=1, maximum=4, value=2, label = 'number of images per prompt', step=1)
            
            
        with gr.Column():
            gallery = gr.Gallery(
                    label="Generate",
                object_fit="contain", height="2048")

            
                
    btn.click(generate, 
              inputs=[text,input_image, guidance_scale, num_images_per_prompt], 
              outputs=gallery)
    

if __name__ == "__main__":
    demo.launch(share=True, server_name = "0.0.0.0")