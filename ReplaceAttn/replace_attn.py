from video_editing_pipeline import VideoEditingPipeline
import cv2
from PIL import Image
import numpy as np
from diffusers import ControlNetModel
import torch


def video_to_frame(video_path: str, interval: int):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    count = 0
    res = []
    while success:
        count += 1
        success, image = vidcap.read()
        if count % interval != 1:
            continue
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image[:, 100:800], (512, 512))
            res.append(image)

    vidcap.release()
    return res


input_video_path = 'woman.mp4'
input_interval = 10
frames = video_to_frame(
    input_video_path, input_interval)
frames = frames[:10]

control_frames = []
# get canny image
for frame in frames:
    np_image = cv2.Canny(frame, 50, 100)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    control_frames.append(canny_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny").to('cuda')

pipeline = VideoEditingPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', controlnet=controlnet).to('cuda')
pipeline.safety_checker = None

generator = torch.manual_seed(0)
frames = [Image.fromarray(frame) for frame in frames]

output_frames = pipeline(images=frames,
                         control_images=control_frames,
                         prompt='a beautiful woman with red hair',
                         num_inference_steps=20,
                         controlnet_conditioning_scale=0.7,
                         strength=0.9,
                         generator=generator,
                         negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')

output_frames[0].save("output.gif", save_all=True,
                      append_images=output_frames[1:], duration=100, loop=0)
