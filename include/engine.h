#pragma once

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#include <vector>

#define VRDX_IMPLEMENTATIONS
#include "vk_radix_sort.h"

#include "gs_core.h"
#include "engine_utils.h"
#include "camera.h"

namespace Core {

typedef struct RendererInfo {
    size_t src_width;
    size_t src_height;
    float scale;
    uint64_t n_gaussians;
} RendererInfo;

class Engine {

public:
    Engine(uint64_t src_width, uint64_t src_height, float scale, std::vector<Core::Point> &points);
    Engine();
    ~Engine();

    void run();

private:

    uint32_t render_width, render_height;
    uint32_t totalGaussians;
    uint32_t gaussBufferCapacity;

    VkDebugUtilsMessengerEXT debugMessenger;

    GLFWwindow* pWindow;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physDev;
    VkDevice device;
    uint32_t graphicsAndComputeFamilyIndex;
    uint32_t presentFamilyIndex;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainImageExtent;

    VkRenderPass renderpass;

    std::vector<VkFence> projFinshedFences;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    VkBuffer gaussianBuffer;
    VkDeviceMemory gaussianBufferMemory;
    VkBuffer projectedGaussianBuffer;
    VkDeviceMemory projectedGaussianBufferMemory;

    uint32_t KVCapacity;
    uint32_t counter;
    VrdxSorter sorter;
    VkBuffer keyBuffer;
    VkDeviceMemory keyBufferMemory;
    VkBuffer valueBuffer;
    VkDeviceMemory valueBufferMemory;
    VkBuffer pingpongBuffer;
    VkDeviceMemory pingpongBufferMemory;
    VkBuffer counterBuffer;
    VkDeviceMemory counterBufferMemory;
    void* pCounter;

    uint32_t num_tiles;
    VkBuffer tileRangeBuffer;
    VkDeviceMemory tileRangeBufferMemory;


    CameraUBO camUBO;
    std::vector<VkBuffer> cameraBuffers; // need to be resized
    std::vector<VkDeviceMemory> cameraBufferMemory;
    std::vector<void*> cameraBufferMapped;

    std::vector<VkImage> offscreenImages;
    std::vector<VkImageView> offscreenImageViews;
    std::vector<VkDeviceMemory> imageMemory;
    // VkSampler sampler;

    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout globalDescriptorSetLayout; // gaussian buffer
    VkDescriptorSetLayout localDescriptorSetLayout; // camera ubo, images
    VkDescriptorSet globalDescriptorSets;
    std::vector<VkDescriptorSet> localDescriptorSets;

    VkPipelineLayout projComputePipelineLayout;
    VkPipeline projComputePipeline;
    VkPipelineLayout rangeComputePipelineLayout;
    VkPipeline rangeComputePipeline;

    void drawFrame();
    void train();

    void initWindow();
    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createSwapchainImageViews();
    void createRenderpass();
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    template <typename UBO>
    void createUniformBuffer(VkBuffer &buffer,
                             VkDeviceMemory &bufferMemory,
                             void* &pData);

    template <typename T>
    void createStorageBuffer(size_t num_elements, VkBuffer &buffer, VkDeviceMemory &bufferMemory);

    template <typename T>
    void createStorageBuffer(std::vector<T> &srcBuffer, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
    
    void createImage(uint32_t width, uint32_t height,
                     VkFormat imageFormat, VkImageUsageFlags imageUsage,
                     VkMemoryPropertyFlags properties, VkImage &image,
                     VkDeviceMemory &imageMemory);
    void createImageView(VkFormat format, VkImage &image, VkImageView &imageView);
    void createSorterAndBuffer();
    void createDescriptorSetLayouts();
    void createDescriptorPool();
    void createDescriptorSets();
    void createComputePipeline(VkPipeline &pipeline, VkPipelineLayout &pipelineLayout,
                               const char* shaderPath,
                               uint32_t pushConstantRangeCount, VkPushConstantRange* pPushConstantRanges,
                               uint32_t setLayoutCount, VkDescriptorSetLayout* pSetLayouts);
    void createCommandPool();
    void createCommandBuffers();
    void createFrameBuffers();
    void createSyncObjects();
    void recordCommandbuffer(VkCommandBuffer &cmdbuf);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
};

}
