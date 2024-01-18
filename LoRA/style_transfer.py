from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

lora_path = '...'
sd_path = 'runwayml/stable-diffusion-v1-5'
controlnet_canny_path = 'lllyasviel/sd-controlnet-canny'

prompt = '1 man, look at right, side face, Ace Attorney, Phoenix Wright, best quality, danganronpa'
neg_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, {multiple people}'
img_path = '...'
init_image = Image.open(img_path).convert("RGB")
init_image = init_image.resize((768, 512))
np_image = np.array(init_image)

# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)
canny_image.save('tmp_edge.png')

controlnet = ControlNetModel.from_pretrained(controlnet_canny_path)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    sd_path, controlnet=controlnet
)
pipe.load_lora_weights(lora_path)

output = pipe(
    prompt=prompt,
    negative_prompt=neg_prompt,
    strength=0.5,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=50,
    image=init_image,
    cross_attention_kwargs={"scale": 1.0},
    control_image=canny_image,
).images[0]
output.save("tmp.png")
