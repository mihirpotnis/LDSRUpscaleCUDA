import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import __init__

# Assuming necessary imports from ldsrlib and related modules
from ldsrlib.LDSR import LDSR
from ldsrlib.ldm.util import instantiate_from_config

import os
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import torch

def load_image(image_path):
    # Ensure the image_path is correct
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")

    # Open the image
    img = Image.open(image_path)
    
    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']
    
    for i in ImageSequence.Iterator(img):
        # Transpose the image based on EXIF data
        i = ImageOps.exif_transpose(i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]
        
        if image.size[0] != w or image.size[1] != h:
            continue
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image, output_mask

# Example usage
image_path = 'example_lowres.png'  # Update this to the correct image path
output_image, output_mask = load_image(image_path)


# Path to the model checkpoint and configuration file
model_ckpt = 'last.ckpt'
config_file = 'ldsrlib/config.yaml'  # Ensure this is the correct path to the config file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Initialize the model
ldsr = LDSR(modelPath=model_ckpt, yamlPath=config_file)

# Load the model from checkpoint
config = OmegaConf.load(config_file)
pl_sd = torch.load(model_ckpt, map_location="cpu")
sd = pl_sd["state_dict"]
model = ldsr.load_model_from_config()

model['model'].load_state_dict(sd, strict=False)
model['model'].to(device)
model['model'].eval()

# Load and prepare the input image
input_image_path = "example_lowres.png"
input_image = Image.open(input_image_path).convert("RGB")

# Normalize and prepare the image for processing
#input_image, w_pad, h_pad = LDSR.normalize_image(input_image)

# Convert image to tensor
input_tensor = torch.tensor(np.array(input_image)).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

# Call the upscale function
steps = "100"  # Example number of steps, adjust based on your needs
pre_downscale = "None"
post_downscale = "None"
downsample_method = "Lanczos"

# Initialize LDSRUpscale
upscaler = __init__.LDSRUpscaler()

# Call the upscale method
ldsr_output = upscaler.upscale('last.ckpt', output_image, steps, pre_downscale, post_downscale, downsample_method)

# Post-process the output
i = 255. * ldsr_output[0].cpu().numpy()
img = Image.fromarray(np.clip(i[0], 0, 255).astype(np.uint8))
#output_image = ldsr_output[0].squeeze().permute(1, 2, 0).cpu().numpy()
#output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
print('out of func')
# Convert numpy array back to image
#output_image_pil = Image.fromarray(output_image)

# Save or display the output image
img.save("image_upscaled.png")
img.show()