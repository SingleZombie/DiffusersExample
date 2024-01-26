from diffusers import StableDiffusionControlNetImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch


class AttnState:
    STORE = 0
    LOAD = 1

    def __init__(self):
        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def timestep(self):
        return self.__timestep

    def set_timestep(self, t):
        self.__timestep = t

    def reset(self):
        self.__state = AttnState.STORE
        self.__timestep = 0

    def to_load(self):
        self.__state = AttnState.LOAD


class CrossFrameAttnProcessor(AttnProcessor):
    """
    Cross frame attention processor. Each frame attends the first frame and previous frame.

    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """

    def __init__(self, attn_state: AttnState):
        super().__init__()
        self.attn_state = attn_state
        self.cur_timestep = 0
        self.first_maps = {}
        self.prev_maps = {}

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, **kwargs):

        if encoder_hidden_states is None:
            # Is self attention

            tot_timestep = self.attn_state.timestep
            if self.attn_state.state == AttnState.STORE:
                self.first_maps[self.cur_timestep] = hidden_states.detach()
                self.prev_maps[self.cur_timestep] = hidden_states.detach()
                res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
            else:
                tmp = hidden_states.detach()
                cross_map = torch.cat(
                    (self.first_maps[self.cur_timestep], self.prev_maps[self.cur_timestep]), dim=1)
                res = super().__call__(attn, hidden_states, cross_map, **kwargs)
                self.prev_maps[self.cur_timestep] = tmp

            self.cur_timestep += 1
            if self.cur_timestep == tot_timestep:
                self.cur_timestep = 0
        else:
            # Is cross attention
            res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        return res


class VideoEditingPipeline(StableDiffusionControlNetImg2ImgPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, controlnet, scheduler,
                         safety_checker, feature_extractor, image_encoder, requires_safety_checker)
        self.attn_state = AttnState()
        attn_processor_dict = {}
        for k in unet.attn_processors.keys():
            if k.startswith("up"):
                attn_processor_dict[k] = CrossFrameAttnProcessor(
                    self.attn_state)
            else:
                attn_processor_dict[k] = AttnProcessor()

        self.unet.set_attn_processor(attn_processor_dict)

    def __call__(self, *args, images=None, control_images=None,  **kwargs):
        self.attn_state.reset()
        self.attn_state.set_timestep(
            int(kwargs['num_inference_steps'] * kwargs['strength']))
        outputs = [super().__call__(
            *args, **kwargs, image=images[0], control_image=control_images[0]).images[0]]
        self.attn_state.to_load()
        for i in range(1, len(images)):
            image = images[i]
            control_image = control_images[i]
            outputs.append(super().__call__(
                *args, **kwargs, image=image, control_image=control_image).images[0])
        return outputs
