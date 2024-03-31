import random
import gradio as gr
from PIL import Image
import torch
import uuid
import numpy as np
from diffusers import AutoPipelineForText2Image, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline
import os
import gdown


# Download weights

weight_ids = {
    "grfield_01.safetensors":"1jhCbuLzvuz73pjRf3mtJwisZiDy4rxuH",
    "garfieldCAT_v2_sdxl_lora.safetensors":"1mQdQ_L32GJYSsKtlQLY-tUDBplr45FDt",
    "garfieldCAT_v3_sdxl_lora.safetensors":"1YktewzpWZTDCbiRdRfqj2Z_AhqGF3Cja",
    "ohwx_cat_64_5.safetensors":"1WuE-YbwzrVa5Da2mwubn01Tk_zv0HVUT",
    "ohwx_v3_sdxl_lora.safetensors":"1JI7mdGDSZzictIqFpeoEWTYp25x0y16a",
    "ohwx_v4_sdxl_lora.safetensors":"1Yu7ocPoxIKB3VKdlCfLmI6pb_ZBmVtGN",
    "ohwx_v4_sdxl_lora.safetensors":"1ruRz3kdEGRxmQ5Ps6BsbEiA8mFD8vyYP",
    "ohwx_v4_sdxl_lora.safetensors":"1uej-XPe2JDSfILkdJmwVExnKMHiZd-bX",
    "qwe_cat_long.safetensors":"1kHDRCvx_yMkz4vOsy-XQAp9kjtFjEbWe",
    "qwe_cat.safetensors":"1R6El6uj90zDmrWBguAI4_-He6UBoS72M",    
}

if len(os.listdir('lora_weights')) == 0:

    for file in weight_ids:
            
        # Define the Google Drive file ID
        file_id = weight_ids[file]

        # Define the output file path
        output_path = f'lora_weight/{file_id}'

        # Download the file
        gdown.download('https://drive.google.com/uc?id=' + file_id, output_path, quiet=False)



model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# we have give lora weights here
pipe = AutoPipelineForText2Image.from_pretrained(model_id, 
                                                torch_dtype=torch.float16, 
                                                variant="fp16")

pipe.to("cuda")

pipe.load_lora_weights('lora_weights/qwe_cat_long.safetensors')
pipe.fuse_lora()


lora_models ={}
trigger_word={}
for i in os.scandir('lora_weights'):
    if i.name != '.gitignore':
        lora_models[i.name] = i.path
        trigger_word[i.name] = i.name.split('_')[0] + ' cat'


def save_img(image_list, prompt):
    results_folder = 'results/'
    os.makedirs(results_folder, exist_ok=True)
    
    for image in image_list:
        image = Image.open(image[0])
        unique_id = uuid.uuid4()
        image.save(f"{results_folder}{unique_id}.jpg")
        # Construct the new file name with UUID
        new_filename = f"{results_folder}{unique_id}.txt"

        # Open the file in write mode
        with open(new_filename, "w") as file:
            # Write the text to the file
            file.write(prompt)


def set_lora_model(lora_name, lora_scale):
    pipe.unfuse_lora(True)
    pipe.unload_lora_weights()
    print(lora_models[lora_name])
    pipe.load_lora_weights(lora_models[lora_name])
    pipe.fuse_lora(lora_scale=lora_scale)
    print('Model swapped')
    
    return trigger_word[lora_name]


def toggle_freeU(freeU_toggle):
    if freeU_toggle:
        print('freeU enabled')
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    else:
        print('freeU disabled')
        pipe.disable_freeu()


def generate(prompt, 
             guidance_scale, 
             num_images_per_prompt, 
             height, 
             width, 
             generator_seed,
             negative_prompt,
             lora_scale
             ):
    
    generator = torch.Generator("cuda").manual_seed(generator_seed)

    image = pipe(prompt=prompt, 
                 negative_prompt=negative_prompt,
                 guidance_scale=guidance_scale,
                 num_images_per_prompt=num_images_per_prompt,
                 height=height,
                 width=width,
                 num_inference_steps=20,
                 generator=generator,
                 cross_attention_kwargs={"scale": lora_scale}
                 ).images
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            
            gallery = gr.Gallery(
                label="Generate",
             object_fit="contain", height="512",
             interactive=False)
            positive_prompt = gr.Textbox(
                    label="Enter Positive Prompt...",
                    value= 'qwe cat'
                    )
            negative_prompt = gr.Textbox(
                    label="Enter Negative Prompt...",
                    value='worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting'
                    )
            
            with gr.Row():
                lora_model_dropdown = gr.Dropdown(list(lora_models.keys()), label='Select LoRA model', value = 'qwe_cat_long.safetensors')
            with gr.Row():
                guidance_scale = gr.Slider(minimum=0, maximum=15, value=9.5, label='guidance scale')
                lora_scale = gr.Slider(minimum=0.1, maximum=1, value=1,step=0.01,label = 'Lora scale')
            with gr.Row():
                num_images_per_prompt = gr.Slider(minimum=1, maximum=4, value=1, step=1, label = 'number of images per prompt')
                generator_seed = gr.Slider(minimum=-1, maximum=100, value=1,step=1,label = 'generator_seed')
            with gr.Row():
                height = gr.Slider(minimum=512, maximum=2048, value=1024, label = 'Image height')
                width = gr.Slider(minimum=512, maximum=2048, value=1024,step=8,label = 'Image width')
                freeu = gr.Checkbox(value=True, label='Toggle FreeU')
    with gr.Row():
        btn = gr.Button("Generate")
        download_btn = gr.Button("Download")
        
    btn.click(generate, 
              inputs=[positive_prompt,
                      guidance_scale,
                      num_images_per_prompt, 
                      height, 
                      width, 
                      generator_seed, 
                      negative_prompt,
                      lora_scale], 
              outputs=gallery)
    download_btn.click(save_img,
                       inputs=[gallery,
                               positive_prompt])
    
    freeu.select(toggle_freeU, freeu)
    
    lora_model_dropdown.select(set_lora_model,
                               [lora_model_dropdown, lora_scale],
                               positive_prompt)
    
if __name__ == "__main__":
    demo.launch(share=True)
