<a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-red?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Ashleigh%20Watson)](https://twitter.com/OutofAi) 
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Alex%20Nasa)](https://twitter.com/banterless_ai)
[![PyPI version](https://badge.fury.io/py/cudacanvas.svg)](https://badge.fury.io/py/cudacanvas)
[![Downloads](https://static.pepy.tech/badge/cudacanvas)](https://pepy.tech/project/cudacanvas)

# cudacanvas
CudaCanvas: Real-time PyTorch Tensor Image Visualisation in CUDA, Eliminating CPU Transfer

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
        #end process if the window is closed
        break


```

And with explicit window creation

```python
import torch
import cudacanvas

noise_image = torch.rand((4, 500, 500), device="cuda")

cudacanvas.set_image(noise_image)
cudacanvas.create_window()

#replace this with you training loop
while (True):

    cudacanvas.render()

    if cudacanvas.should_close():
        #end process if the window is closed
        break


```


# Installation
Before instllation make sure you have torch with cuda support already installed on your machine

We aligned pytorch and cuda version with our package the supporting packages are torch (2.2.2 or 2.1.2) and (11.8 or 12.1)

Identify your current torch and cuda version

```python
import torch
torch.__version__
```

Depending on your torch and cuda you can install the relevant cudacanvas package, for the latest 2.2.2+cu121 you can simply download the latest package
```
pip install cudacanvas
```
For other torch and cuda packages replace the +cu from the torch version with .post and prefix it with 1. which is the current cudacanvas version to generate the relevant package to install so for example for 2.1.2+cu181 the Cudacanvas package you require
is 1.2.1.2.post181

```
pip install cudacanvas==1.2.1.2.post118
```

# Support
Also support my channel ☕ ☕ : https://www.buymeacoffee.com/outofai
