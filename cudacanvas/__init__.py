try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Torch with CUDA is required for this module!")
except ImportError:
    raise ImportError("Torch is not installed!")

try:
    import pkg_resources
    
    __version__ = pkg_resources.get_distribution("cudacanvas").version
    # Fetching the CUDA version from PyTorch and formatting the version string
    torch_version = torch.__version__.split('+')[0].replace(".", "")  # Gets the base version of torch, e.g., '2.2.2'
    cuda_version = torch.version.cuda.replace(".", "")  # Gets CUDA version, e.g., '118'
    version_base = torch_version  # Base version now uses the PyTorch version

    full_version = "1.0.1" + ".post" + version_base + cuda_version

    if ( not __version__ == full_version): 
        print(f'You currently installed Cudacanvas v{__version__} which does not match your torch+cuda version ({torch.__version__})')
        print(f'try:  \033[92mpip install cudacanvas=={full_version} --force-reinstall\033[0m')
        print('to see if the required version is available')
               
except pkg_resources.DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

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
