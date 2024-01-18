import torch
from inversion_pipeline import InversionPipeline

lora_path = 'ckpt/mountain.safetensor'
lora_path2 = 'ckpt/mountain_up.safetensor'
sd_path = 'runwayml/stable-diffusion-v1-5'


@torch.no_grad()
def slerp(p0, p1, fract_mixing: float):
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()

    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


pipeline: InversionPipeline = InversionPipeline.from_pretrained(
    sd_path).to("cuda")
pipeline.load_lora_weights(lora_path, adapter_name='a')
pipeline.load_lora_weights(lora_path2, adapter_name='b')

img1_path = 'dataset/mountain/mountain.jpg'
img2_path = 'dataset/mountain_up/mountain_up.jpg'
prompt = 'mountain'
latent1 = pipeline.inverse(img1_path, prompt, 50, guidance_scale=1)
latent2 = pipeline.inverse(img2_path, prompt, 50, guidance_scale=1)
n_frames = 10
images = []
for i in range(n_frames + 1):
    alpha = i / n_frames
    pipeline.set_adapters(["a", "b"], adapter_weights=[1 - alpha, alpha])
    latent = slerp(latent1, latent2, alpha)
    output = pipeline(prompt=prompt, latents=latent,
                      guidance_scale=1.0).images[0]
    images.append(output)

images[0].save("output.gif", save_all=True,
               append_images=images[1:], duration=100, loop=0)
