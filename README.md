<a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-orange?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Ashleigh%20Watson)](https://twitter.com/OutofAi) 
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Alex%20Nasa)](https://twitter.com/banterless_ai)

# cudacanvas
CudaCanvas: Real-time PyTorch Tensor Visualisation in CUDA, Eliminating CPU Transfer

CudaCanvas is a simple Python module that eliminates CPU transfer for Pytorch tensors for displaying and rendering images in the training or evaluation phase, ideal for machine learning scientists and engineers. 

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

Identify your current torch and cuda version, cudacanvas currently only supports torch 2.1.2 and cuda (11.8 or 12.1)

```python
import torch
torch.__version__
```

### CUDA 12.1
If you are running torch 2.1.2 with Cuda 12.1 (2.1.2+cu121) you can download it straight from pypi by running
```
pip install cudacanvas
```

### CUDA 11.8
If you are running torch 2.1.2 with Cuda 11.8 (2.1.2+cu118) you can run this script
```
pip install cudacanvas --find-links https://github.com/OutofAi/cudacanvas/wiki/cu118
```
or manaully download the latest wheel releases from https://github.com/OutofAi/cudacanvas/releases/

# Support
Also support my channel ☕ ☕ : https://www.buymeacoffee.com/outofai
