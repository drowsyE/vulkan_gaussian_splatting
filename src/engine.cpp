// #define NDEBUG

#define VRDX_IMPLEMENTATION
#include "engine.h"
#include "engine_utils.h"
#include "gs_core.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#define MAX_FRAME_IN_FLIGHT 2
int currentFrame = 0;

const int DEFAULT_WIDTH = 1200;
const int DEFAULT_HEIGHT = 1000;

namespace Core {

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif

// Function pointer for Push Descriptors
extern "C" PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ptr =
        nullptr;

std::vector<const char *> validationLayer{"VK_LAYER_KHRONOS_validation"};

std::vector<const char *> deviceExtensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME,
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, // for radix sort
#if __APPLE__
            "VK_KHR_portability_subset"
#endif
};

void Engine::run() {
    double previousTime = glfwGetTime();
    double lastFrameTime = previousTime;
    int frameCount = 0;

    while (!glfwWindowShouldClose(pWindow)) {
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastFrameTime);
        lastFrameTime = currentTime;
        frameCount++;
        if (currentTime - previousTime >= 1.0) {
            std::cout << "FPS: " << frameCount << std::endl;
            frameCount = 0;
            previousTime = currentTime;
        }

        glfwPollEvents();
        updateCameraUBO(deltaTime);
        drawFrame();
    }
}

void Engine::drawFrame() {
    VkCommandBuffer cmdbuf = computeCommandBuffers[currentFrame];
    VkCommandBuffer cmdbuf2 = computeCommandBuffers2[currentFrame];
    VkCommandBuffer graphicsCmdbuf = graphicsCommandBuffers[currentFrame];

    // --- Synchronization & Resets ---
    vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
    vkWaitForFences(device, 1, &graphicsInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &graphicsInFlightFences[currentFrame]);

    vkResetCommandBuffer(cmdbuf, 0);
    vkResetCommandBuffer(cmdbuf2, 0);
    vkResetCommandBuffer(graphicsCmdbuf, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                                                                     1};

    // --- Pass A (Projection & Argpass) ---
    vkBeginCommandBuffer(cmdbuf, &beginInfo);
    vkCmdFillBuffer(cmdbuf, counterBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmdbuf, tileRangeBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmdbuf, projectedGaussianBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmdbuf, keyBuffer, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmdbuf, valueBuffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFF);

    VkMemoryBarrier initBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    initBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    initBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdbuf,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &initBarrier, 0, nullptr, 0, nullptr);

    uint32_t pushData[2] = {totalGaussians, KVCapacity};
    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, projComputePipeline);
    vkCmdPushConstants(cmdbuf, projComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushData), pushData);
    VkDescriptorSet bindDescriptorSets[] = {globalDescriptorSets, localDescriptorSets[currentFrame]};
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, projComputePipelineLayout, 0, 2, bindDescriptorSets, 0, nullptr);
    vkCmdDispatch(cmdbuf, (totalGaussians + 255) / 256, 1, 1);

    VkMemoryBarrier midPassABarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    midPassABarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    midPassABarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdbuf,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &midPassABarrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpassComputePipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpassComputePipelineLayout, 0, 2, bindDescriptorSets, 0, nullptr);
    vkCmdPushConstants(cmdbuf, argpassComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &KVCapacity);
    vkCmdDispatch(cmdbuf, 1, 1, 1);
    vkEndCommandBuffer(cmdbuf);

    VkSubmitInfo submitInfoA{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfoA.commandBufferCount = 1;
    submitInfoA.pCommandBuffers = &cmdbuf;
    submitInfoA.signalSemaphoreCount = 1;
    submitInfoA.pSignalSemaphores = &projectionFinishedSemaphores[currentFrame];
    vkQueueSubmit(computeQueue, 1, &submitInfoA, VK_NULL_HANDLE);

    // --- Pass B (Sort, Range, Raster) ---
    vkBeginCommandBuffer(cmdbuf2, &beginInfo);
    vrdxCmdSortKeyValueIndirect(cmdbuf2, sorter, KVCapacity,
                                counterBuffer, 0, keyBuffer, 0,
                                valueBuffer, 0, pingpongBuffer, 0, VK_NULL_HANDLE, 0);

    VkMemoryBarrier postSortBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    postSortBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    postSortBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmdbuf2,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                        0, 1, &postSortBarrier, 0, nullptr, 0, nullptr);

    vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rangeComputePipelineLayout, 0, 1, &globalDescriptorSets, 0, nullptr);
    vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,rangeComputePipeline);
    vkCmdPushConstants(cmdbuf2, rangeComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &num_tiles);
    vkCmdDispatchIndirect(cmdbuf2, indirectArgsBuffer, 0);

    VkImageMemoryBarrier imageToGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageToGeneralBarrier.image = offscreenImages[currentFrame];
    imageToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageToGeneralBarrier.srcAccessMask = VK_ACCESS_NONE;
    imageToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageToGeneralBarrier.subresourceRange = subresourceRange;

    VkMemoryBarrier postRangeBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    postRangeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postRangeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmdbuf2,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &postRangeBarrier, 0, nullptr, 1, &imageToGeneralBarrier);

    vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rasterComputePipeline);
    vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rasterComputePipelineLayout, 0, 2, bindDescriptorSets, 0, nullptr);
    vkCmdDispatch(cmdbuf2, (render_width + 15) / 16, (render_height + 15) / 16, 1);
    vkEndCommandBuffer(cmdbuf2);

    VkSubmitInfo submitInfoB{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkPipelineStageFlags computeWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    submitInfoB.commandBufferCount = 1;
    submitInfoB.pCommandBuffers = &cmdbuf2;
    submitInfoB.waitSemaphoreCount = 1;
    submitInfoB.pWaitSemaphores = &projectionFinishedSemaphores[currentFrame];
    submitInfoB.pWaitDstStageMask = computeWaitStages;
    submitInfoB.signalSemaphoreCount = 1;
    submitInfoB.pSignalSemaphores = &computeFinishedSemaphore[currentFrame];
    vkQueueSubmit(computeQueue, 1, &submitInfoB, computeInFlightFences[currentFrame]);

#ifndef NDEBUG
    static int verifyCounter = 0;
    if (verifyCounter++ % 100 == 0) {
        vkQueueWaitIdle(computeQueue);
        verifyRadixSort(false);
    }
#endif

    // --- Graphics Passthrough ---
    uint32_t imageIndex;
    vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
						  imageAvailableSemaphore[currentFrame], VK_NULL_HANDLE, &imageIndex);

    vkBeginCommandBuffer(graphicsCmdbuf, &beginInfo);
    VkImageMemoryBarrier offscreenToTransferBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    offscreenToTransferBarrier.image = offscreenImages[currentFrame];
    offscreenToTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    offscreenToTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    offscreenToTransferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    offscreenToTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    offscreenToTransferBarrier.subresourceRange = subresourceRange;
    vkCmdPipelineBarrier(graphicsCmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
						nullptr, 1, &offscreenToTransferBarrier);

    VkImageMemoryBarrier swapchainToTransferBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapchainToTransferBarrier.image = swapchainImages[imageIndex];
    swapchainToTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchainToTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapchainToTransferBarrier.srcAccessMask = VK_ACCESS_NONE;
    swapchainToTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapchainToTransferBarrier.subresourceRange = subresourceRange;
    vkCmdPipelineBarrier(graphicsCmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
						VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
						nullptr, 1, &swapchainToTransferBarrier);

    VkImageBlit blitRegion{};
    blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.srcOffsets[0] = {0, 0, 0};
    blitRegion.srcOffsets[1] = {(int)render_width, (int)render_height, 1};
    blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.dstOffsets[0] = {0, 0, 0};
    blitRegion.dstOffsets[1] = {(int)swapchainImageExtent.width, (int)swapchainImageExtent.height, 1};
    vkCmdBlitImage(
            graphicsCmdbuf, offscreenImages[currentFrame],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

    VkImageMemoryBarrier offscreenBackToGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    offscreenBackToGeneralBarrier.image = offscreenImages[currentFrame];
    offscreenBackToGeneralBarrier.oldLayout =VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    offscreenBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    offscreenBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    offscreenBackToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    offscreenBackToGeneralBarrier.subresourceRange = subresourceRange;
    vkCmdPipelineBarrier(graphicsCmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
						nullptr, 1, &offscreenBackToGeneralBarrier);

    VkImageMemoryBarrier swapchainToPresentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapchainToPresentBarrier.image = swapchainImages[imageIndex];
    swapchainToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapchainToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapchainToPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapchainToPresentBarrier.dstAccessMask = VK_ACCESS_NONE;
    swapchainToPresentBarrier.subresourceRange = subresourceRange;
    vkCmdPipelineBarrier(graphicsCmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
						nullptr, 1, &swapchainToPresentBarrier);
    vkEndCommandBuffer(graphicsCmdbuf);

    VkSemaphore graphicsWaitSemaphores[] = {computeFinishedSemaphore[currentFrame], imageAvailableSemaphore[currentFrame]};
    VkPipelineStageFlags graphicsWaitStages[] = {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    VkSubmitInfo graphicsSubmitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    graphicsSubmitInfo.commandBufferCount = 1;
    graphicsSubmitInfo.pCommandBuffers = &graphicsCmdbuf;
    graphicsSubmitInfo.waitSemaphoreCount = 2;
    graphicsSubmitInfo.pWaitSemaphores = graphicsWaitSemaphores;
    graphicsSubmitInfo.pWaitDstStageMask = graphicsWaitStages;
    graphicsSubmitInfo.signalSemaphoreCount = 1;
    graphicsSubmitInfo.pSignalSemaphores = &renderFinishedSemaphore[imageIndex];
    vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, graphicsInFlightFences[currentFrame]);

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore[imageIndex];
    vkQueuePresentKHR(presentQueue, &presentInfo);

    currentFrame = (currentFrame + 1) % MAX_FRAME_IN_FLIGHT;
}

void Engine::train() {}

Engine::Engine(uint64_t src_width, uint64_t src_height, float scale, std::vector<Core::Point> &points) {

    initWindow();
    createInstance();
    setupDebugMessenger();
    glfwCreateWindowSurface(instance, pWindow, nullptr, &surface);
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createSwapchainImageViews();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();

    render_width = src_width * scale;
    render_height = src_height * scale;
    totalGaussians = points.size();
    maxGaussians = totalGaussians << 1;
    std::vector<Gaussian3D> src_tmp = gaussianFromPoints(points, totalGaussians, maxGaussians);

    createStorageBuffer<Gaussian3D>(src_tmp, gaussianBuffer, gaussianBufferMemory);
    createStorageBuffer<Gaussian2D>(maxGaussians, projectedGaussianBuffer, projectedGaussianBufferMemory);

    int n_tiles_row = (render_height + 15) >> 4;
    int n_tiles_col = (render_width + 15) >> 4;
    num_tiles = n_tiles_row * n_tiles_col;
    createStorageBuffer<TileRange>(num_tiles, tileRangeBuffer, tileRangeBufferMemory);
    createBuffer(sizeof(uint32_t) * 3,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
				VK_BUFFER_USAGE_TRANSFER_DST_BIT |
				VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indirectArgsBuffer,
				indirectArgsBufferMemory);

    cameraBuffers.resize(MAX_FRAME_IN_FLIGHT);
    cameraBufferMemory.resize(MAX_FRAME_IN_FLIGHT);
    cameraBufferMapped.resize(MAX_FRAME_IN_FLIGHT);
    offscreenImages.resize(MAX_FRAME_IN_FLIGHT);
    offscreenImageViews.resize(MAX_FRAME_IN_FLIGHT);
    imageMemory.resize(MAX_FRAME_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        createUniformBuffer<CameraUBO>(cameraBuffers[i], cameraBufferMemory[i], cameraBufferMapped[i]);
        createImage(render_width, render_height, VK_FORMAT_R8G8B8A8_UNORM,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
					VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreenImages[i],
					imageMemory[i]);
        createImageView(VK_FORMAT_R8G8B8A8_UNORM, offscreenImages[i], offscreenImageViews[i]);
    }
    createSorterAndBuffer();
    createDescriptorSetLayouts();
    createDescriptorPool();
    createDescriptorSets();

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(uint32_t) * 2; // totalGaussians + kvCapacity
    std::array<VkDescriptorSetLayout, 2> setLayout = {globalDescriptorSetLayout, localDescriptorSetLayout};
    createComputePipeline(projComputePipeline, projComputePipelineLayout,
						"../shader/spv/proj.spv", 1, &push, setLayout.size(), setLayout.data());
    VkPushConstantRange rangePush{};
    rangePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    rangePush.offset = 0;
    rangePush.size = sizeof(uint32_t); // num_tiles
    createComputePipeline(rangeComputePipeline, rangeComputePipelineLayout,
						"../shader/spv/range.spv", 1, &rangePush, 1, &globalDescriptorSetLayout);
    createComputePipeline(rasterComputePipeline, rasterComputePipelineLayout,
						"../shader/spv/raster.spv", 0, nullptr, setLayout.size(), setLayout.data());

    VkPushConstantRange argpassPush{};
    argpassPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    argpassPush.offset = 0;
    argpassPush.size = sizeof(uint32_t); // kvCapacity
    createComputePipeline(argpassComputePipeline, argpassComputePipelineLayout,
						"../shader/spv/argpass.spv", 1, &argpassPush, setLayout.size(), setLayout.data());
    // createRenderpass();
}

Engine::~Engine() {
    vkDeviceWaitIdle(device);
    // vkDestroyRenderPass(device, renderpass, nullptr);
    vkDestroyPipeline(device, argpassComputePipeline, nullptr);
    vkDestroyPipeline(device, rasterComputePipeline, nullptr);
    vkDestroyPipeline(device, rangeComputePipeline, nullptr);
    vkDestroyPipeline(device, projComputePipeline, nullptr);
    vkDestroyPipelineLayout(device, argpassComputePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, rasterComputePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, rangeComputePipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, projComputePipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, localDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, globalDescriptorSetLayout, nullptr);

    vrdxDestroySorter(sorter);

    vkDestroyBuffer(device, keyBuffer, nullptr);
    vkDestroyBuffer(device, valueBuffer, nullptr);
    vkDestroyBuffer(device, counterBuffer, nullptr);
    vkDestroyBuffer(device, pingpongBuffer, nullptr);
    vkDestroyBuffer(device, tileRangeBuffer, nullptr);
    vkDestroyBuffer(device, indirectArgsBuffer, nullptr);
    vkDestroyBuffer(device, gaussianBuffer, nullptr);
    vkDestroyBuffer(device, projectedGaussianBuffer, nullptr);

    vkFreeMemory(device, keyBufferMemory, nullptr);
    vkFreeMemory(device, valueBufferMemory, nullptr);
    vkFreeMemory(device, counterBufferMemory, nullptr);
    vkFreeMemory(device, pingpongBufferMemory, nullptr);
    vkFreeMemory(device, tileRangeBufferMemory, nullptr);
    vkFreeMemory(device, indirectArgsBufferMemory, nullptr);
    vkFreeMemory(device, gaussianBufferMemory, nullptr);
    vkFreeMemory(device, projectedGaussianBufferMemory, nullptr);

    for (int i = 0; i < swapchainImages.size(); i++) {
        vkDestroySemaphore(device, renderFinishedSemaphore[i], nullptr);
    }

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, cameraBuffers[i], nullptr);
        vkDestroyImage(device, offscreenImages[i], nullptr);
        vkDestroyImageView(device, offscreenImageViews[i], nullptr);
        vkFreeMemory(device, cameraBufferMemory[i], nullptr);
        vkFreeMemory(device, imageMemory[i], nullptr);
        vkDestroyFence(device, computeInFlightFences[i], nullptr);
        vkDestroyFence(device, graphicsInFlightFences[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore[i], nullptr);
        vkDestroySemaphore(device, computeFinishedSemaphore[i], nullptr);
        vkDestroySemaphore(device, projectionFinishedSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    for (int i = 0; i < swapchainImageViews.size(); i++) {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    if (enableValidationLayers) {
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(pWindow);
    glfwTerminate();
}

void Engine::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    pWindow = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, "3DGS", nullptr, nullptr);
}

void Engine::createInstance() {

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = nullptr;
    appInfo.pApplicationName = "3DGS";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "no engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t extCount;
    const char **ext = glfwGetRequiredInstanceExtensions(&extCount);
    std::vector<const char *> instanceExtensions(ext, ext + extCount);

#ifdef __APPLE__
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    if (enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkDebugUtilsMessengerCreateInfoEXT messengerInfo{};
    populateDebugMessenger(messengerInfo);

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    instanceInfo.pNext = nullptr;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (enableValidationLayers) {
        instanceInfo.enabledLayerCount = validationLayer.size();
        instanceInfo.ppEnabledLayerNames = validationLayer.data();
        instanceInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&messengerInfo;
    } else {
        instanceInfo.enabledLayerCount = 0;
        instanceInfo.ppEnabledLayerNames = nullptr;
    }

#ifdef __APPLE__
    instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    vkCreateInstance(&instanceInfo, nullptr, &instance);
}

void Engine::setupDebugMessenger() {
    if (!enableValidationLayers)
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessenger(createInfo);

    createDebugUtilsMessenger(instance, &createInfo, nullptr, &debugMessenger);
}

void Engine::pickPhysicalDevice() {

    uint32_t n_physDev;
    vkEnumeratePhysicalDevices(instance, &n_physDev, nullptr);
    std::vector<VkPhysicalDevice> physDevs(n_physDev);
    vkEnumeratePhysicalDevices(instance, &n_physDev, physDevs.data());

    for (const VkPhysicalDevice &device : physDevs) {
        VkPhysicalDeviceProperties deviceProps;
        vkGetPhysicalDeviceProperties(device, &deviceProps);

        uint32_t n_queue;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &n_queue, nullptr);
        std::vector<VkQueueFamilyProperties> deviceQueueProps(n_queue);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &n_queue, deviceQueueProps.data());

        graphicsAndComputeFamilyIndex = -1;
        presentFamilyIndex = -1;
        for (int i = 0; i < deviceQueueProps.size(); i++) {
            if (deviceQueueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
				deviceQueueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                graphicsAndComputeFamilyIndex = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                presentFamilyIndex = i;
            }

            if (graphicsAndComputeFamilyIndex != -1 && presentFamilyIndex != -1) {
                physDev = device;
                printf("\n[Info] | Device selected : %s\n", deviceProps.deviceName);
                printf("[Info] | Graphics Family : %d, Present Family : %d\n", graphicsAndComputeFamilyIndex, presentFamilyIndex);
                return;
            }
        }
    }

    throw std::runtime_error("Failed to find compatible physical device!\n");
}

void Engine::createLogicalDevice() {

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priorities = 1.0f;

    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priorities;
    queueInfos.push_back(queueInfo);
    if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
        queueInfo.queueFamilyIndex = presentFamilyIndex;
        queueInfos.push_back(queueInfo);
    }

    VkPhysicalDeviceFeatures supportedDeviceFeatures;
    vkGetPhysicalDeviceFeatures(physDev, &supportedDeviceFeatures);

    VkPhysicalDeviceFeatures deviceFeatures{};
    if (supportedDeviceFeatures.shaderInt64) {
        deviceFeatures.shaderInt64 = VK_TRUE;
    } else {
        throw std::runtime_error("This device doesn't supports shaderInt64 feature!");
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = queueInfos.size();
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceInfo.pEnabledFeatures = &deviceFeatures;
    if (enableValidationLayers) {
        deviceInfo.enabledLayerCount = validationLayer.size();
        deviceInfo.ppEnabledLayerNames = validationLayer.data();
    } else {
        deviceInfo.enabledLayerCount = 0;
        deviceInfo.ppEnabledLayerNames = nullptr;
    }

    vkCreateDevice(physDev, &deviceInfo, nullptr, &device);

    // Load push descriptor function pointer
    vkCmdPushDescriptorSetKHR_ptr = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    if (vkCmdPushDescriptorSetKHR_ptr == nullptr) {
        fprintf(stderr, "[Critical] vkCmdPushDescriptorSetKHR_ptr is NULL! Radix sort WILL NOT WORK.\n");
    } else {
        fprintf(stderr, "[Info] vkCmdPushDescriptorSetKHR_ptr successfully loaded.\n");
    }
    if (!vkCmdPushDescriptorSetKHR_ptr) {
        throw std::runtime_error("Found no vkCmdPushDescriptorSetKHR function pointer!");
    }

    vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &computeQueue);
    vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);
}

void Engine::createSwapchain() {
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, surface, &surfaceCaps);

    uint32_t minImageCount;
    if (surfaceCaps.minImageCount + 1 > surfaceCaps.maxImageCount) {
        minImageCount = surfaceCaps.maxImageCount;
    } else {
        minImageCount = surfaceCaps.minImageCount + 1;
    }

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFormats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCount, surfaceFormats.data());

    VkFormat format = surfaceFormats[0].format;
    VkColorSpaceKHR colorspace = surfaceFormats[0].colorSpace;
    for (const VkSurfaceFormatKHR &fmt : surfaceFormats) {
        if (fmt.format == VK_FORMAT_R8G8B8A8_SRGB && fmt.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
            format = VK_FORMAT_R8G8B8A8_SRGB;
            colorspace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
            break;
        }
    }
    swapchainImageFormat = format;

    VkExtent2D imageExtent;
    if (surfaceCaps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        imageExtent = surfaceCaps.currentExtent;
    } else {
        int width, height;
        glfwGetWindowSize(pWindow, &width, &height);

        imageExtent.width = std::clamp<int>(width, surfaceCaps.minImageExtent.width, surfaceCaps.maxImageExtent.width);
        imageExtent.height = std::clamp<int>(width, surfaceCaps.minImageExtent.height, surfaceCaps.maxImageExtent.height);
    }
    swapchainImageExtent = imageExtent;

    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<uint32_t> queueFamilyIndices = {graphicsAndComputeFamilyIndex};
    if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
        sharingMode = VK_SHARING_MODE_CONCURRENT;
        queueFamilyIndices.push_back(presentFamilyIndex);
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &presentModeCount, presentModes.data());

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (const VkPresentModeKHR &mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
    }

    VkSwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = minImageCount;
    swapchainInfo.imageFormat = format;
    swapchainInfo.imageColorSpace = colorspace;
    swapchainInfo.imageExtent = imageExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainInfo.imageSharingMode = sharingMode;
    swapchainInfo.queueFamilyIndexCount = queueFamilyIndices.size();
    swapchainInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    swapchainInfo.presentMode = presentMode;
    swapchainInfo.preTransform = surfaceCaps.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.clipped = VK_TRUE;
    vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain);

    vkGetSwapchainImagesKHR(device, swapchain, &minImageCount, nullptr);
    swapchainImages.resize(minImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &minImageCount, swapchainImages.data());
}

void Engine::createSwapchainImageViews() {
    swapchainImageViews.resize(swapchainImages.size());

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.format = swapchainImageFormat;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
						.g = VK_COMPONENT_SWIZZLE_IDENTITY,
						.b = VK_COMPONENT_SWIZZLE_IDENTITY,
						.a = VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
								.baseMipLevel = 0,
								.levelCount = 1,
								.baseArrayLayer = 0,
								.layerCount = 1};

    for (int i = 0; i < swapchainImages.size(); i++) {
        viewInfo.image = swapchainImages[i];
        vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[i]);
    }
}

// void Engine::createRenderpass() {

// 	VkAttachmentDescription colorAtt{};
// 	colorAtt.format = swapchainImageFormat;
// 	colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
// 	colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_NONE; // 이미 compute shader에서 그렸기 때문 
// 	colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
// 	colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
// 	colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
// 	colorAtt.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
// 	colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

// 	VkAttachmentReference colorAttRef{};
// 	colorAttRef.attachment = 0;
// 	colorAttRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

// 	VkSubpassDescription subpass{};
// 	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
// 	subpass.colorAttachmentCount = 1;
// 	subpass.pColorAttachments = &colorAttRef;

// 	VkSubpassDependency dependency{};
// 	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
// 	dependency.dstSubpass = 0;
// 	dependency.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
// 	dependency.srcAccessMask = 0;
// 	dependency.dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
// 	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

// 	VkRenderPassCreateInfo renderInfo{};
// 	renderInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
// 	renderInfo.attachmentCount = 1;
// 	renderInfo.pAttachments = &colorAtt;
// 	renderInfo.subpassCount = 1;
// 	renderInfo.pSubpasses = &subpass;
// 	renderInfo.dependencyCount = 1;
// 	renderInfo.pDependencies = &dependency;

// 	vkCreateRenderPass(device, &renderInfo, nullptr, &renderpass);
// }

void Engine::createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;

    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
}

void Engine::createCommandBuffers() {
    computeCommandBuffers.resize(MAX_FRAME_IN_FLIGHT);
    computeCommandBuffers2.resize(MAX_FRAME_IN_FLIGHT);
    graphicsCommandBuffers.resize(MAX_FRAME_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAME_IN_FLIGHT;
    allocInfo.commandPool = commandPool;
    vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data());
    vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers2.data());
    vkAllocateCommandBuffers(device, &allocInfo, graphicsCommandBuffers.data());
}

void Engine::createSorterAndBuffer() {
    VrdxSorterCreateInfo sorterInfo{.device = device,
									.physicalDevice = physDev,
									.pipelineCache = VK_NULL_HANDLE};
    vrdxCreateSorter(&sorterInfo, &sorter);

    KVCapacity = maxGaussians << 6; // static pre-allocation pool (increased to fix blocks)

    // VkDeviceSize keyBufferSize = totalKVCount * sizeof(uint64_t);
    // VkDeviceSize valueBufferSize = totalKVCount * sizeof(uint32_t);
    VkDeviceSize counterBufferSize = sizeof(uint32_t);

    VrdxSorterStorageRequirements reqs;
    vrdxGetSorterKeyValueStorageRequirements(sorter, KVCapacity, &reqs);

    // key/value 버퍼는 최소 KVCapacity * sizeof(uint32_t) 이상이어야 함
    // reqs.size는 sort 내부 스토리지(pingpong) 전용 크기이므로 별도로 계산
    VkDeviceSize kvBufferSize =
            std::max(reqs.size, (VkDeviceSize)(KVCapacity * sizeof(uint32_t)));
    VkBufferUsageFlags kvUsage = reqs.usage |
								VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
								VK_BUFFER_USAGE_TRANSFER_DST_BIT |
								VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    createBuffer(kvBufferSize, kvUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, keyBuffer, keyBufferMemory);
    createBuffer(kvBufferSize, kvUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, valueBuffer, valueBufferMemory);
    createBuffer(reqs.size, reqs.usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pingpongBuffer, pingpongBufferMemory);
    // counterBuffer needs TRANSFER_DST to be reset every frame, and TRANSFER_SRC for sort
    createBuffer(
            counterBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			counterBuffer, counterBufferMemory);
}

void Engine::createSyncObjects() {
    computeInFlightFences.resize(MAX_FRAME_IN_FLIGHT);
    graphicsInFlightFences.resize(MAX_FRAME_IN_FLIGHT);
    computeFinishedSemaphore.resize(MAX_FRAME_IN_FLIGHT);
    projectionFinishedSemaphores.resize(MAX_FRAME_IN_FLIGHT);
    imageAvailableSemaphore.resize(MAX_FRAME_IN_FLIGHT);
    renderFinishedSemaphore.resize(swapchainImages.size());

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaInfo{};
    semaInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
        vkCreateSemaphore(device, &semaInfo, nullptr, &computeFinishedSemaphore[i]);
        vkCreateSemaphore(device, &semaInfo, nullptr, &projectionFinishedSemaphores[i]);
        vkCreateSemaphore(device, &semaInfo, nullptr, &imageAvailableSemaphore[i]);
        vkCreateFence(device, &fenceInfo, nullptr, &graphicsInFlightFences[i]);
        vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]);
    }
    for (int i = 0; i < swapchainImages.size(); i++) {
        vkCreateSemaphore(device, &semaInfo, nullptr, &renderFinishedSemaphore[i]);
    }
}

void Engine::createDescriptorSetLayouts() {

    // 1. global descriptor
    std::array<VkDescriptorSetLayoutBinding, 7> globalBindings{};
    globalBindings[0].binding = 0;
    globalBindings[0].descriptorCount = 1;
    globalBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[0].pImmutableSamplers = nullptr;
    globalBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[1].binding = 1;
    globalBindings[1].descriptorCount = 1;
    globalBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[1].pImmutableSamplers = nullptr;
    globalBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[2].binding = 2; // Keys
    globalBindings[2].descriptorCount = 1;
    globalBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[2].pImmutableSamplers = nullptr;
    globalBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[3].binding = 3; // Values
    globalBindings[3].descriptorCount = 1;
    globalBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[3].pImmutableSamplers = nullptr;
    globalBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[4].binding = 4; // Counter
    globalBindings[4].descriptorCount = 1;
    globalBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[4].pImmutableSamplers = nullptr;
    globalBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[5].binding = 5; // Tile ranges
    globalBindings[5].descriptorCount = 1;
    globalBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[5].pImmutableSamplers = nullptr;
    globalBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    globalBindings[6].binding = 6; // Dispatch xyz
    globalBindings[6].descriptorCount = 1;
    globalBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    globalBindings[6].pImmutableSamplers = nullptr;
    globalBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
    setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutInfo.bindingCount = globalBindings.size();
    setLayoutInfo.pBindings = globalBindings.data();
    vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &globalDescriptorSetLayout);

    // 2. local descriptor
    std::array<VkDescriptorSetLayoutBinding, 2> localBindings{};
    // UBO for camera
    localBindings[0].binding = 0;
    localBindings[0].descriptorCount = 1;
    localBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    localBindings[0].pImmutableSamplers = nullptr;
    localBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Output image
    localBindings[1].binding = 1;
    localBindings[1].descriptorCount = 1;
    localBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    localBindings[1].pImmutableSamplers = nullptr;
    localBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    setLayoutInfo.bindingCount = localBindings.size();
    setLayoutInfo.pBindings = localBindings.data();
    vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &localDescriptorSetLayout);
}

void Engine::createDescriptorPool() {

    std::array<VkDescriptorPoolSize, 3> poolSizes;
    poolSizes[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 7};
    poolSizes[1] = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_FRAME_IN_FLIGHT};
    poolSizes[2] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = MAX_FRAME_IN_FLIGHT};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = MAX_FRAME_IN_FLIGHT + 1;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void Engine::createDescriptorSets() {

    // printf("DEBUG | global descriptor set layout : %p\n",
    // (void*)globalDescriptorSetLayout); printf("DEBUG | local descriptor set
    // layout : %p\n", (void*)localDescriptorSetLayout);

    // 1. global descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &globalDescriptorSetLayout;

    // if (vkAllocateDescriptorSets(device, &allocInfo, &globalDescriptorSets) !=
    //         VK_SUCCESS) {
    //     throw std::runtime_error("failed to allocate global descriptor set!");
    // }
    vkAllocateDescriptorSets(device, &allocInfo, &globalDescriptorSets);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = gaussianBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(Gaussian3D) * maxGaussians;

    VkDescriptorBufferInfo bufferInfo1{};
    bufferInfo1.buffer = projectedGaussianBuffer;
    bufferInfo1.offset = 0;
    bufferInfo1.range = sizeof(Gaussian2D) * maxGaussians;

    VkDescriptorBufferInfo bufferInfo2{};
    bufferInfo2.buffer = keyBuffer;
    bufferInfo2.offset = 0;
    bufferInfo2.range = sizeof(uint32_t) * KVCapacity;

    VkDescriptorBufferInfo bufferInfo3{};
    bufferInfo3.buffer = valueBuffer;
    bufferInfo3.offset = 0;
    bufferInfo3.range = sizeof(uint32_t) * KVCapacity;

    VkDescriptorBufferInfo bufferInfo4{};
    bufferInfo4.buffer = counterBuffer;
    bufferInfo4.offset = 0;
    bufferInfo4.range = sizeof(uint32_t);

    VkDescriptorBufferInfo bufferInfo5{};
    bufferInfo5.buffer = tileRangeBuffer;
    bufferInfo5.offset = 0;
    bufferInfo5.range = sizeof(TileRange) * num_tiles;

    VkDescriptorBufferInfo bufferInfo6{};
    bufferInfo6.buffer = indirectArgsBuffer;
    bufferInfo6.offset = 0;
    bufferInfo6.range = sizeof(uint32_t) * 3;

    std::array<VkWriteDescriptorSet, 7> descriptorWriteGlobal{};
    descriptorWriteGlobal[0] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[1] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo1,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[2] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo2,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[3] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo3,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[4] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 4,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo4,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[5] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 5,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo5,
            .pTexelBufferView = nullptr,
    };

    descriptorWriteGlobal[6] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = globalDescriptorSets,
            .dstBinding = 6,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &bufferInfo6,
            .pTexelBufferView = nullptr,
    };

    vkUpdateDescriptorSets(device, descriptorWriteGlobal.size(), descriptorWriteGlobal.data(), 0, nullptr);

    // 2. local descriptor set
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAME_IN_FLIGHT, localDescriptorSetLayout);

    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    localDescriptorSets.resize(MAX_FRAME_IN_FLIGHT);
    vkAllocateDescriptorSets(device, &allocInfo, localDescriptorSets.data());

    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {

        VkDescriptorBufferInfo camInfo{};
        camInfo.buffer = cameraBuffers[i];
        camInfo.offset = 0;
        camInfo.range = sizeof(CameraUBO);

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView = offscreenImageViews[i];
        imgInfo.sampler = nullptr;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> descriptorWritesLocal;
        descriptorWritesLocal[0] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = localDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &camInfo,
                .pTexelBufferView = nullptr,
        };

        descriptorWritesLocal[1] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = localDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &imgInfo,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(device, descriptorWritesLocal.size(),
                                                     descriptorWritesLocal.data(), 0, nullptr);
    }
}

void Engine::createComputePipeline(VkPipeline &pipeline,
								VkPipelineLayout &pipelineLayout,
								const char *shaderPath,
								uint32_t pushConstantRangeCount,
								VkPushConstantRange *pPushConstantRanges,
								uint32_t setLayoutCount,
								VkDescriptorSetLayout *pSetLayouts) {

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = pushConstantRangeCount;
    pipelineLayoutInfo.pPushConstantRanges = pPushConstantRanges;
    pipelineLayoutInfo.setLayoutCount = setLayoutCount;
    pipelineLayoutInfo.pSetLayouts = pSetLayouts;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkShaderModule shaderModule = createShader(device, shaderPath);
    VkPipelineShaderStageCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderInfo.module = shaderModule;
    shaderInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage = shaderInfo;
    vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &pipeline);

    vkDestroyShaderModule(device, shaderModule, nullptr);
}

// void Renderer::createComputePipeline(const char* shaderPath,
//									uint32_t pushConstantRangeCount,
//									VkPushConstantRange*
//									pPushConstantRanges, uint32_t
//									setLayoutCount, VkDescriptorSetLayout*
//									pSetLayouts) {

//         VkPushConstantRange push{};
//         push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
//         push.offset = 0;
//         push.size = sizeof(int);

//         std::array<VkDescriptorSetLayout, 2> setLayout =
//         {globalDescriptorSetLayout, localDescriptorSetLayout};

//         VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
//         pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
//         pipelineLayoutInfo.pushConstantRangeCount = 1;
//         pipelineLayoutInfo.pPushConstantRanges = &push;
//         pipelineLayoutInfo.setLayoutCount = setLayout.size();
//         pipelineLayoutInfo.pSetLayouts = setLayout.data();
//         vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
//         &computePipelineLayout);

//         VkShaderModule renderShaderModule = createShader(device,
//         "../shader/spv/proj.spv"); VkPipelineShaderStageCreateInfo
//         renderShaderInfo{}; renderShaderInfo.sType =
//         VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
//         renderShaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//         renderShaderInfo.module = renderShaderModule;
//         renderShaderInfo.pName = "main";

//         VkComputePipelineCreateInfo pipelineInfo{};
//         pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
//         pipelineInfo.layout = computePipelineLayout;
//         pipelineInfo.stage = renderShaderInfo;
//         vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr,
//         &computePipeline);

//         vkDestroyShaderModule(device, renderShaderModule, nullptr);
// }

uint32_t Engine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) &&
                (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

void Engine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
						VkMemoryPropertyFlags properties,
						VkBuffer &buffer,
						VkDeviceMemory &bufferMemory) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
    vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void Engine::createImage(uint32_t width, uint32_t height,
						VkFormat imageFormat,
						VkImageUsageFlags imageUsage,
						VkMemoryPropertyFlags properties,
						VkImage &image, VkDeviceMemory &imageMemory) {

    VkImageCreateInfo imgCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = imageFormat,
            .extent = {.width = width, .height = height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = imageUsage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    vkCreateImage(device, &imgCreateInfo, nullptr, &image);

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
    vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);

    vkBindImageMemory(device, image, imageMemory, 0);
}

void Engine::createImageView(VkFormat format, VkImage &image, VkImageView &imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.format = format;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
						.g = VK_COMPONENT_SWIZZLE_IDENTITY,
						.b = VK_COMPONENT_SWIZZLE_IDENTITY,
						.a = VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
								.baseMipLevel = 0,
								.levelCount = 1,
								.baseArrayLayer = 0,
								.layerCount = 1};
    vkCreateImageView(device, &viewInfo, nullptr, &imageView);
}

template <typename UBO>
void Engine::createUniformBuffer(VkBuffer &buffer, VkDeviceMemory &bufferMemory, void *&pData) {

    VkDeviceSize size = sizeof(UBO);
    createBuffer(size,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				buffer, bufferMemory);

    // persistent mapping
    vkMapMemory(device, bufferMemory, 0, size, 0, &pData);
}

template <typename T>
void Engine::createStorageBuffer(size_t num_elements, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
    VkDeviceSize size = sizeof(T) * num_elements;

    createBuffer(size,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				buffer, bufferMemory);
}

template <typename T>
void Engine::createStorageBuffer(std::vector<T> &srcBuffer, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {

    VkDeviceSize size = sizeof(T) * srcBuffer.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    void *pData;

    createBuffer(size,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

    vkMapMemory(device, stagingBufferMemory, 0, size, 0, &pData);
    memcpy(pData, srcBuffer.data(), size);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				buffer, bufferMemory);

    VkBufferCopy region{};
    region.size = size;

    VkCommandBuffer cmdbuf = beginSingleTimeCommands();
    vkCmdCopyBuffer(cmdbuf, stagingBuffer, buffer, 1, &region);
    endSingleTimeCommands(cmdbuf);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

VkCommandBuffer Engine::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    allocInfo.commandPool = commandPool;

    VkCommandBuffer cmdbuf;
    vkAllocateCommandBuffers(device, &allocInfo, &cmdbuf);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdbuf, &beginInfo);

    return cmdbuf;
}

void Engine::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
};

void Engine::updateCameraUBO(float deltaTime) {
    float moveSpeed = 3.0f * deltaTime;
    float rotSpeed = 60.0f * deltaTime;

    if (glfwGetKey(pWindow, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += moveSpeed * cameraFront;
    if (glfwGetKey(pWindow, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= moveSpeed * cameraFront;
	if (glfwGetKey(pWindow, GLFW_KEY_SPACE) == GLFW_PRESS)
		cameraPos += moveSpeed * cameraUp;
	if (glfwGetKey(pWindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cameraPos -= moveSpeed * cameraUp;
    if (glfwGetKey(pWindow, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * moveSpeed;
    if (glfwGetKey(pWindow, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * moveSpeed;

    if (glfwGetKey(pWindow, GLFW_KEY_UP) == GLFW_PRESS)
        pitch -= rotSpeed;
    if (glfwGetKey(pWindow, GLFW_KEY_DOWN) == GLFW_PRESS)
        pitch += rotSpeed;
    if (glfwGetKey(pWindow, GLFW_KEY_LEFT) == GLFW_PRESS)
        yaw += rotSpeed;
    if (glfwGetKey(pWindow, GLFW_KEY_RIGHT) == GLFW_PRESS)
        yaw -= rotSpeed;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    updateCamera(camUBO, cameraPos, cameraPos + cameraFront, render_width, render_height, cameraUp);
    memcpy(cameraBufferMapped[currentFrame], &camUBO, sizeof(CameraUBO));
}

void Engine::setCameraFromColmap(const Image &image) {
    glm::quat q(image.q[0], image.q[1], image.q[2], image.q[3]);
    glm::mat3 R_cw = glm::mat3_cast(q);
    glm::vec3 t_cw(image.t[0], image.t[1], image.t[2]);

    // camera pose C = -R_cw^T * t_cw
    glm::mat3 R_wc = glm::transpose(R_cw);
    cameraPos = -R_wc * t_cw;

    // Colmap convention: camera space is X-right, Y-down, Z-forward
    // World space Up vector of the camera is R_wc * (0, -1, 0)
    cameraFront = glm::normalize(R_wc * glm::vec3(0.0f, 0.0f, 1.0f));
    cameraUp = glm::normalize(R_wc * glm::vec3(0.0f, -1.0f, 0.0f));

    // 시야 확보를 위해 카메라를 살짝 뒤로 뺌
    cameraPos -= cameraFront * 2.0f;

    pitch = glm::degrees(asin(cameraFront.y));
    yaw = glm::degrees(atan2(cameraFront.z, cameraFront.x));

    // Force update immediately
    updateCamera(camUBO, cameraPos, cameraPos + cameraFront, render_width, render_height, cameraUp);
    for (int i = 0; i < MAX_FRAME_IN_FLIGHT; ++i) {
        memcpy(cameraBufferMapped[i], &camUBO, sizeof(CameraUBO));
    }
}

void Engine::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

void Engine::verifyRadixSort(bool preSort) {
    // 1. Read element count from counterBuffer
    uint32_t count = 0;
    {
        VkBuffer stagingCount;
        VkDeviceMemory stagingCountMemory;
        createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 stagingCount, stagingCountMemory);
        copyBuffer(counterBuffer, stagingCount, sizeof(uint32_t));
        void *pCount;
        vkMapMemory(device, stagingCountMemory, 0, sizeof(uint32_t), 0, &pCount);
        count = *(uint32_t *)pCount;
        vkUnmapMemory(device, stagingCountMemory);
        vkDestroyBuffer(device, stagingCount, nullptr);
        vkFreeMemory(device, stagingCountMemory, nullptr);
    }

    fprintf(stderr, "[Verify] Sorting count reported by GPU: %u\n", count);
    if (count == 0)
        return;
    if (count > KVCapacity)
        count = KVCapacity;

    auto checkBuffer = [&](VkBuffer buf, const char *name) {
        VkBuffer stagingKeys;
        VkDeviceMemory stagingKeysMemory;
        VkDeviceSize bufferSize = count * sizeof(uint32_t);
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 stagingKeys, stagingKeysMemory);
        copyBuffer(buf, stagingKeys, bufferSize);

        void *pKeys;
        vkMapMemory(device, stagingKeysMemory, 0, bufferSize, 0, &pKeys);
        uint32_t *keys = (uint32_t *)pKeys;

        bool sorted = true;
        for (uint32_t i = 0; i < count - 1; i++) {
            if (keys[i] > keys[i + 1]) {
                sorted = false;
                break;
            }
        }
        if (sorted) {
            fprintf(stderr,
					"[Verify SUCCESS] Buffer '%s' is PERFECTLY SORTED (%u keys)!\n",
					name, count);
            fprintf(stderr,
					"    Sample[0..4] Keys:     %u, %u, %u, %u, %u\n", keys[0],
					keys[1], keys[2], keys[3], keys[4]);
            // Also maybe sample some values if we had them mapped
        } else {
            // Find where it failed exactly
            uint32_t failIdx = 0;
            for (uint32_t i = 0; i < count - 1; i++) {
                if (keys[i] > keys[i + 1]) {
                    failIdx = i;
                    break;
                }
            }
            fprintf(stderr,
					"[Verify Fail] Buffer '%s' NOT sorted at index %u: %u > %u "
					"(Total: %u)\n",
					name, failIdx, keys[failIdx], keys[failIdx + 1], count);
            fprintf(stderr,
					"    Sample around failure: [%u]=%u, [%u]=%u, [%u]=%u\n",
					failIdx > 0 ? failIdx - 1 : 0,
					failIdx > 0 ? keys[failIdx - 1] : 0, failIdx, keys[failIdx],
					failIdx + 1, keys[failIdx + 1]);
        }

        vkUnmapMemory(device, stagingKeysMemory);
        vkDestroyBuffer(device, stagingKeys, nullptr);
        vkFreeMemory(device, stagingKeysMemory, nullptr);
        return sorted;
    };

    checkBuffer(keyBuffer, "keyBuffer (Original)");
}
} // namespace Core
