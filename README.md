<a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-red?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Out%20of%20AI)](https://twitter.com/OutofAi)
[![PyPI version](https://badge.fury.io/py/cudacanvas.svg)](https://badge.fury.io/py/cudacanvas)
[![Downloads](https://static.pepy.tech/badge/cudacanvas)](https://pepy.tech/project/cudacanvas)

![image](https://github.com/OutofAi/cudacanvas/assets/145302363/94f1ba88-0991-4690-b09b-7be480ee34ec)


# cudacanvas : PyTorch Tensor Image Display in CUDA
(Real-time PyTorch Tensor Image Visualisation in CUDA, Eliminating CPU Transfer)

CudaCanvas is a simple Python module that eliminates CPU transfer for Pytorch tensors for displaying and rendering images in the training or evaluation phase, ideal for machine learning scientists and engineers. 

Simplified version that directly displays the image without explicit window creation (cudacanvas >= v1.0.1)

```python
import torch
import cudacanvas


#REPLACE THIS with you training loop
while (True):

    #REPLACE THIS with you training code and generation of data
    noise_image = torch.rand((4, 500, 500), device="cuda")

    #Visualise your data in real-time
    cudacanvas.im_show(noise_image)

    #OPTIONAL: Terminate training when the window is closed
    if cudacanvas.should_close():
        cudacanvas.clean_up()
        #end process if the window is closed
        break


```

You can visualise the latent of Stable Diffusion during sampling in real-time whilst waiting for the steps to finish

```python

import warnings
warnings.filterwarnings("ignore")
from diffusers import StableDiffusionPipeline
import torch
import cudacanvas

def display_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = image - image.min()
        image = image / image.max()
    
    cudacanvas.im_show(image.squeeze(0))
    
    if cudacanvas.should_close():
        cudacanvas.clean_up()
        pipe._interrupt = True
    
    return callback_kwargs

pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image = pipeline(
    prompt="A croissant shaped like a cute bear.",
    negative_prompt="Deformed, ugly, bad anatomy",
    callback_on_step_end=display_tensors,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]

cudacanvas.clean_up()

```


# Installation
Before instllation make sure you have torch with cuda support already installed on your machine

We aligned pytorch and cuda version with our package the supporting packages are torch (2.0.1, 2.1.2 and 2.2.2) and (11.8 and 12.1)

Identify your current torch and cuda version

```python
import torch
torch.__version__
```

Depending on your torch and cuda you can install the relevant cudacanvas package, for the latest one matching the latest pytorch package you can simply download the latest package
```
pip install cudacanvas
```
For other torch and cuda packages put the torch and cuda version after that cudacanvas version for example for 2.1.2+cu118 the Cudacanvas package you require
is 1.0.1.post212118

```
pip install cudacanvas==1.0.1.post212118 --force-reinstall
```

# Support
Also support my channel ☕ ☕ : https://www.buymeacoffee.com/outofai
