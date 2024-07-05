import sys
from os import path
sys.path.insert(0, path.dirname(__file__))
from ldsrlib.LDSR import LDSR
from os import path, listdir
sys.path.insert(0, path.dirname(__file__))
import torch

class LDSRModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_list = [f for f in listdir("upscale_models") if f.endswith('.ckpt')]
        candidates = [name for name in model_list if 'last.ckpt' in name]
        default_path = candidates[0] if candidates else 'last.ckpt'

        return {
            "required": {
                "model": (model_list, {'default': default_path}),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load"

    CATEGORY = "Flowty LDSR"

    def load(self, model):
        model_path = path.join("upscale_models", model)
        model = LDSR.load_model_from_path(model_path)
        model['model'].to(torch.device("cuda"))
        return (model, )


class LDSRUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),
                "steps": (["25", "50", "100", "250", "500", "1000"], {"default": "100"}),
                "pre_downscale": (['None', '1/2', '1/4'], {"default": "None"}),
                "post_downscale": (['None', 'Original Size', '1/2', '1/4'], {"default": "None"}),
                "downsample_method": (['Nearest', 'Lanczos'], {"default": "Lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    CATEGORY = "Flowty LDSR"

    def upscale(self, upscale_model, images, steps, pre_downscale="None", post_downscale="None", downsample_method="Lanczos"):
        def progress_bar(steps):
            for i in range(steps):
                yield i

        pbar = progress_bar(int(steps))

        ldsr = LDSR(model=upscale_model, torchdevice=torch.device("cuda"), on_progress=lambda i: next(pbar))

        outputs = []

        for image in images:
            image = image.to(torch.device("cuda"))
            outputs.append(ldsr.superResolution(image, int(steps), pre_downscale, post_downscale, downsample_method))

        return (torch.stack(outputs).to("cpu"),)


class LDSRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        model_list = [f for f in listdir("upscale_models") if f.endswith('.ckpt')]
        candidates = [name for name in model_list if 'last.ckpt' in name]
        default_path = candidates[0] if candidates else 'last.ckpt'

        return {
            "required": {
                "model": (model_list, {'default': default_path}),
                "images": ("IMAGE",),
                "steps": (["25", "50", "100", "250", "500", "1000"], {"default": "100"}),
                "pre_downscale": (['None', '1/2', '1/4'], {"default": "None"}),
                "post_downscale": (['None', 'Original Size', '1/2', '1/4'], {"default": "None"}),
                "downsample_method": (['Nearest', 'Lanczos'], {"default": "Lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    CATEGORY = "Flowty LDSR"

    def upscale(self, model, images, steps, pre_downscale="None", post_downscale="None", downsample_method="Lanczos"):
        #model_path = path.join("upscale_models", model)
        model_path=model
        def progress_bar(steps):
            for i in range(steps):
                yield i

        pbar = progress_bar(int(steps))

        ldsr = LDSR(modelPath=model_path, torchdevice=torch.device("cuda"), on_progress=lambda i: next(pbar))

        outputs = []

        for image in images:
            image = image.to(torch.device("cuda"))
            outputs.append(ldsr.superResolution(image, int(steps), pre_downscale, post_downscale, downsample_method))

        return (torch.stack(outputs).to("cpu"),)


NODE_CLASS_MAPPINGS = {
    "LDSRUpscaler": LDSRUpscaler,
    "LDSRModelLoader": LDSRModelLoader,
    "LDSRUpscale": LDSRUpscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LDSRUpscaler": "LDSR Upscale (all-in-one)",
    "LDSRModelLoader": "Load LDSR Model",
    "LDSRUpscale": "LDSR Upscale"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
