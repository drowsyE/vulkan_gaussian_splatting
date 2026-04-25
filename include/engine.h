#pragma once

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>
#include <vector>

#include "vk_radix_sort.h"

#include "camera.h"
#include "engine_utils.h"
#include "gs_core.h"

namespace Core {

class Engine {

public:
  Engine(uint64_t src_width, uint64_t src_height, float scale,
         std::vector<Core::Point> &points);
  Engine(uint64_t src_width, uint64_t src_height, float scale,
         std::vector<Core::Gaussian3D> &gaussians);
  Engine();

  ~Engine();

  void run();
  void train(std::vector<Image> &images, std::vector<Camera> &cameras, int32_t iterations,
             float learning_rate = 1e-4, float beta1 = 0.9, float beta2 = 0.999,
             float eps = 1e-8, float lambda = 0.2);
  void setCameraFromColmap(const Image &image, const Camera *camera = nullptr);

private:
  static constexpr int MAX_FRAME_IN_FLIGHT = 2;
  int currentFrame = 0;

  uint32_t render_width, render_height;
  uint32_t totalGaussians;
  uint32_t maxGaussians; // capacity

  VkDebugUtilsMessengerEXT debugMessenger;

  GLFWwindow *pWindow;
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

  std::vector<VkSemaphore> computeFinishedSemaphore;
  std::vector<VkSemaphore> imageAvailableSemaphore;
  std::vector<VkSemaphore> renderFinishedSemaphore;
  std::vector<VkSemaphore> projectionFinishedSemaphores;
  std::vector<VkFence> computeInFlightFences;
  std::vector<VkFence> graphicsInFlightFences;

  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> computeCommandBuffers;
  std::vector<VkCommandBuffer> computeCommandBuffers2;
  std::vector<VkCommandBuffer> graphicsCommandBuffers;

  uint32_t KVCapacity;
  uint32_t counter;

  VrdxSorter sorter;

  VkBuffer gaussianBuffer;
  VkBuffer projectedGaussianBuffer;
  VkBuffer keyBuffer;
  VkBuffer valueBuffer;
  VkBuffer pingpongBuffer;
  VkBuffer counterBuffer;
  VkBuffer indirectArgsBuffer;
  VkBuffer gaussianCountBuffer;

  VkDeviceMemory gaussianBufferMemory;
  VkDeviceMemory projectedGaussianBufferMemory;
  VkDeviceMemory keyBufferMemory;
  VkDeviceMemory valueBufferMemory;
  VkDeviceMemory pingpongBufferMemory;
  VkDeviceMemory counterBufferMemory;
  VkDeviceMemory indirectArgsBufferMemory;
  VkDeviceMemory gaussianCountBufferMemory;

  uint32_t num_tiles;
  VkBuffer tileRangeBuffer;
  VkDeviceMemory tileRangeBufferMemory;

  CameraUBO camUBO;
  glm::vec3 cameraPos{0.0f, 0.0f, 5.0f};
  glm::vec3 cameraFront{0.0f, 0.0f, -1.0f};
  glm::vec3 cameraUp{0.0f, -1.0f, 0.0f};
  float yaw{-90.0f};
  float pitch{0.0f};
  float roll{0.0f};
  std::vector<VkBuffer> cameraBuffers; // need to be resized
  std::vector<VkDeviceMemory> cameraBufferMemory;
  std::vector<void *> cameraBufferMapped;

  std::vector<VkImage> offscreenImages;
  std::vector<VkImageView> offscreenImageViews;
  std::vector<VkDeviceMemory> imageMemory;

  VkDescriptorPool descriptorPool;
  VkDescriptorSetLayout globalDescriptorSetLayout; // gaussian buffer
  VkDescriptorSetLayout localDescriptorSetLayout;  // camera ubo, images
  VkDescriptorSet globalDescriptorSets;
  std::vector<VkDescriptorSet> localDescriptorSets;

  VkPipelineLayout projComputePipelineLayout;
  VkPipeline projComputePipeline;
  VkPipelineLayout rangeComputePipelineLayout;
  VkPipeline rangeComputePipeline;
  VkPipelineLayout rasterComputePipelineLayout;
  VkPipeline rasterComputePipeline;
  VkPipelineLayout argpassComputePipelineLayout;
  VkPipeline argpassComputePipeline;
  VkPipelineLayout argpass2PipelineLayout;
  VkPipeline argpass2Pipeline;

  void drawFrame();

  void initWindow();
  void createInstance();
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapchain();
  void createSwapchainImageViews();
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory);
  template <typename UBO>
  void createUniformBuffer(VkBuffer &buffer, VkDeviceMemory &bufferMemory,
                           void *&pData);

  template <typename T>
  void createStorageBuffer(size_t num_elements, VkBuffer &buffer,
                           VkDeviceMemory &bufferMemory);

  template <typename T>
  void createStorageBuffer(std::vector<T> &srcBuffer, VkBuffer &buffer,
                           VkDeviceMemory &bufferMemory);

  template <typename T>
  void createStorageBuffer(std::vector<T> &srcBuffer, VkBufferUsageFlags usage,
                           VkBuffer &buffer, VkDeviceMemory &bufferMemory);

  void createImage(uint32_t width, uint32_t height, VkFormat imageFormat,
                   VkImageUsageFlags imageUsage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory);
  void createImageView(VkFormat format, VkImage &image, VkImageView &imageView);
  void createSorterAndBuffer();
  void createDescriptorSetLayouts();
  void createDescriptorPool();
  void createDescriptorSets();
  void createComputePipeline(VkPipeline &pipeline,
                             VkPipelineLayout &pipelineLayout,
                             const char *shaderPath,
                             uint32_t pushConstantRangeCount,
                             VkPushConstantRange *pPushConstantRanges,
                             uint32_t setLayoutCount,
                             VkDescriptorSetLayout *pSetLayouts);
  void createCommandPool();
  void createCommandBuffers();
  void verifyRadixSort(bool preSort = false);
  void createSyncObjects();

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties);
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);
  void endSingleTimeComputeCommands(VkCommandBuffer commandBuffer);
  void updateCameraUBO(float deltaTime);
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height);
  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout);
  float calculateSceneExtent(std::vector<Image> &images);
};

} // namespace Core
