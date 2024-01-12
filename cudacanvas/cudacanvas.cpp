#include <GLFW/glfw3.h>
#include <torch/extension.h>
#include <stdexcept>
#include <cuda_gl_interop.h>

namespace py = pybind11;

class CudaGLStreamer
{
public:
    CudaGLStreamer()
    {
    }

    ~CudaGLStreamer()
    {
    }

    void setImage(torch::Tensor image)
    {
        if (!image.is_cuda()) {
            throw std::runtime_error("Input tensor is not on CUDA. Please provide a CUDA tensor.");
        }

        // Permute the tensor from CHW to HWC format
        image = image.permute({1, 2, 0});

        setImageHWC(image);
    }

    void setImageHWC(torch::Tensor image)
    {
        if (!image.is_cuda()) {
            throw std::runtime_error("Input tensor is not on CUDA. Please provide a CUDA tensor.");
        }

        if (image.size(2) == 3) {  // Assuming the format is [H, W, C]
            // Create an alpha channel filled with 255 (fully opaque)
            auto alpha = torch::full({image.size(0), image.size(1), 1}, 255, image.options());

            // Concatenate the alpha channel to the original image to get 4 channels
            image = torch::cat({image, alpha}, 2);
        }
        source_image = image.mul(255).to(torch::kU8);
    }


    bool shouldClose()
    {
        const bool is_closed = glfwWindowShouldClose(window);

        if (is_closed) {
            cleanUp();
        }
        return is_closed;
    }

    void setTitle(const char *title)
    {
        title_set = true;
        glfwSetWindowTitle(window, title);
    }

    void createWindow()
    {
        // Initialize GLFW in the constructor
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        window = glfwCreateWindow(source_image.size(1), source_image.size(0), "Display Noise Image", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        if (!title_set) {
            glfwSetWindowTitle(window, "cudacanvas");
        }

        glfwMakeContextCurrent(window);

        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // 0x812F is typically the value for GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, source_image.size(1), source_image.size(0), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        GLenum glError = glGetError();
        if (glError != GL_NO_ERROR)
        {
            fprintf(stderr, "OpenGL error before registering texture: %d\n", glError);
        }

        cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaGraphicsGLRegisterImage failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        glEnable(GL_TEXTURE_2D);
    }

    void imShow(torch::Tensor image)
    {
        setImage(image);

        if (!window) {
            createWindow();
        }
        render();
    }

    void render()
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        cudaError_t cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        cudaArray_t texturePtr;
        cudaStatus = cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaResource, 0, 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaGraphicsSubResourceGetMappedArray failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        cudaStatus = cudaMemcpy2DToArray(texturePtr, 0, 0, source_image.data_ptr(), source_image.size(1) * sizeof(uint8_t) * 4, source_image.size(1) * sizeof(uint8_t) * 4, source_image.size(0), cudaMemcpyDeviceToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy2DToArray failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, 0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    void cleanUp()
    {
        if (cudaResource)
            cudaGraphicsUnregisterResource(cudaResource);

        if (textureID)
            glDeleteTextures(1, &textureID);

        if (window)
            glfwDestroyWindow(window);

        glfwTerminate();
    }

private:
    torch::Tensor source_image;
    cudaGraphicsResource_t cudaResource;
    GLuint textureID = 0;
    GLFWwindow *window = nullptr;
    bool title_set = false;
};

PYBIND11_MODULE(cudaGLStream, m)
{
    py::class_<CudaGLStreamer>(m, "CudaGLStreamer")
        .def(py::init<>())
        .def("set_image", &CudaGLStreamer::setImage)
        .def("set_title", &CudaGLStreamer::setTitle)
        .def("create_window", &CudaGLStreamer::createWindow)
        .def("im_show", &CudaGLStreamer::imShow)
        .def("render", &CudaGLStreamer::render)
        .def("should_close", &CudaGLStreamer::shouldClose)
        .def("clean_up", &CudaGLStreamer::cleanUp);
}
