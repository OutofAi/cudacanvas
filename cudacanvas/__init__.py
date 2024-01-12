try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Torch with CUDA is required for this module!")
except ImportError:
    raise ImportError("Torch is not installed!")

try:
    import glfw
except ImportError:
    raise ImportError("GLFW is not installed!")

from .cudaGLStream import CudaGLStreamer

# Create a global instance of CudaGLStreamer
_streamer = CudaGLStreamer()

# Expose methods of CudaGLStreamer at the package level
set_image = _streamer.set_image
set_title = _streamer.set_title
create_window = _streamer.create_window
im_show = _streamer.im_show
render = _streamer.render
should_close = _streamer.should_close
clean_up = _streamer.clean_up