#define NDEBUG

#define VRDX_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "engine.h"
#include "adam.h"
#include "engine_utils.h"
#include "glm/gtc/matrix_transform.hpp"
#include "gs_core.h"
#include "stb_image.h"
#include "stb_image_resize2.h"

#include <iostream>
#include <cmath>
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
extern "C" PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ptr = nullptr;

std::vector<const char *> validationLayer{"VK_LAYER_KHRONOS_validation"};

std::vector<const char *> deviceExtensions {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME,
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
	
	uint32_t n_tiles_col = (render_width + 15) / 16;
	uint32_t n_tiles_row = (render_height + 15) / 16;

  	// --- Synchronization & Resets ---
	vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
	vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
	vkWaitForFences(device, 1, &graphicsInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
	vkResetFences(device, 1, &graphicsInFlightFences[currentFrame]);

	vkResetCommandBuffer(cmdbuf, 0);
	vkResetCommandBuffer(cmdbuf2, 0);
	vkResetCommandBuffer(graphicsCmdbuf, 0);

	VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

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

	vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpass2Pipeline);
	vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
							argpass2PipelineLayout,
							0, 1, &globalDescriptorSets, 0, nullptr);
	vkCmdDispatch(cmdbuf, 1, 1, 1);

	VkMemoryBarrier postArgpassBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	postArgpassBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	postArgpassBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
	vkCmdPipelineBarrier(cmdbuf,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
						0, 1, &postArgpassBarrier, 0, nullptr, 0, nullptr);

	// gaussianCountBuffer is pre-initialized in generator
	vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, projComputePipeline);
	ProjPush projPush{KVCapacity, (uint32_t)n_tiles_col, (uint32_t)n_tiles_row};
	vkCmdPushConstants(cmdbuf, projComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProjPush),
						&projPush);
	VkDescriptorSet bindDescriptorSets[] = {globalDescriptorSets, localDescriptorSets[currentFrame]};
	vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
							projComputePipelineLayout,
							0, 2, bindDescriptorSets, 0, nullptr);
	vkCmdDispatchIndirect(cmdbuf, indirectArgsBuffer, 0);

	VkMemoryBarrier midPassABarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	midPassABarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	midPassABarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmdbuf,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 1, &midPassABarrier, 0, nullptr, 0, nullptr);

	vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpassComputePipeline);
	vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
							argpassComputePipelineLayout, 0, 2,
							bindDescriptorSets, 0, nullptr);
	vkCmdPushConstants(cmdbuf, argpassComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
						&KVCapacity);
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
	vrdxCmdSortKeyValueIndirect(cmdbuf2, sorter, KVCapacity, counterBuffer, 0,
								keyBuffer, 0, valueBuffer, 0, pingpongBuffer, 0,
								VK_NULL_HANDLE, 0);

	VkMemoryBarrier postSortBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	postSortBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	postSortBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
	vkCmdPipelineBarrier(cmdbuf2,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
						0, 1, &postSortBarrier, 0, nullptr, 0, nullptr);

	vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
							rangeComputePipelineLayout, 0, 1,
							&globalDescriptorSets, 0, nullptr);
	vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rangeComputePipeline);
	vkCmdPushConstants(cmdbuf2, rangeComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
						&num_tiles);
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
	ProjPush rasterPush{KVCapacity, (uint32_t)n_tiles_col, (uint32_t)n_tiles_row};
	vkCmdPushConstants(cmdbuf2, rasterComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProjPush),
						&rasterPush);
	vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
							rasterComputePipelineLayout, 0, 2, bindDescriptorSets,
							0, nullptr);
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
						imageAvailableSemaphore[currentFrame], VK_NULL_HANDLE,
						&imageIndex);

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
	vkCmdPipelineBarrier(graphicsCmdbuf,
						VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
						VK_PIPELINE_STAGE_TRANSFER_BIT,
						0, 0, nullptr, 0, nullptr, 1, &swapchainToTransferBarrier);

	VkImageBlit blitRegion{};
	blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blitRegion.srcOffsets[0] = {0, 0, 0};
	blitRegion.srcOffsets[1] = {(int)render_width, (int)render_height, 1};
	blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	blitRegion.dstOffsets[0] = {0, 0, 0};
	blitRegion.dstOffsets[1] = {(int)swapchainImageExtent.width, (int)swapchainImageExtent.height, 1};
	vkCmdBlitImage(graphicsCmdbuf,
					offscreenImages[currentFrame], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1, &blitRegion, VK_FILTER_LINEAR);

	VkImageMemoryBarrier offscreenBackToGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	offscreenBackToGeneralBarrier.image = offscreenImages[currentFrame];
	offscreenBackToGeneralBarrier.oldLayout =VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	offscreenBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	offscreenBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	offscreenBackToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	offscreenBackToGeneralBarrier.subresourceRange = subresourceRange;
	vkCmdPipelineBarrier(graphicsCmdbuf,
						VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 0, nullptr, 0, nullptr, 1, &offscreenBackToGeneralBarrier);

	VkImageMemoryBarrier swapchainToPresentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	swapchainToPresentBarrier.image = swapchainImages[imageIndex];
	swapchainToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	swapchainToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	swapchainToPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	swapchainToPresentBarrier.dstAccessMask = VK_ACCESS_NONE;
	swapchainToPresentBarrier.subresourceRange = subresourceRange;
	vkCmdPipelineBarrier(graphicsCmdbuf,
						VK_PIPELINE_STAGE_TRANSFER_BIT,
						VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
						0, 0, nullptr, 0, nullptr, 1, &swapchainToPresentBarrier);
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

// lambda -> Loss : L1 loss * (1-lambda) + SSIM loss * lambda
void Engine::train(std::vector<Image> &images, std::vector<Camera> &cameras, int32_t iterations,
					float lr, float beta1, float beta2, float eps, float lambda) {

	if (lambda < 0. || lambda > 1.) {
		throw std::runtime_error("lambda must be between zero and one!");
	}

	vkDeviceWaitIdle(device);
	
	uint32_t n_tiles_col = (render_width + 15) / 16;
	uint32_t n_tiles_row = (render_height + 15) / 16;

	std::vector<std::vector<uint8_t>> gtImageCache(images.size());

	VkImage outContribution; // image which records the index of lastest gaussian contributed to pixel color
	VkImage gtImage;
	VkImage curSSIMStats;   // mean_x var_x cov_xy, T_last
	VkImage gtSSIMStats;    // mean_y var_y
	VkImage tempSSIMStats;  // mean_x var_x cov_xy mean_y
	VkImage tempSSIMStats2; // var_y

	VkDeviceMemory outContributionMemory;
	VkDeviceMemory gtImageMemory;
	VkDeviceMemory curSSIMStatsMemory;
	VkDeviceMemory gtSSIMStatsMemory;
	VkDeviceMemory tempSSIMStatsMemory;
	VkDeviceMemory tempSSIMStatsMemory2;

	VkBuffer contributionCountBuffer;
	VkDeviceMemory contributionCountBufferMemory;

	VkBuffer gaussianGradBuffer;
	VkDeviceMemory gaussianGradBufferMemory;
	// 16 floats (64 bytes) per gaussian: pos3d[3] pad scale[3] opacity rot[4]
	// color[3] pad
	createBuffer(sizeof(float) * 16 * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				gaussianGradBuffer, gaussianGradBufferMemory);

	VkBuffer gradPos2dBuffer;
	VkDeviceMemory gradPos2dBufferMemory;
	createBuffer(sizeof(float) * 2 * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				gradPos2dBuffer, gradPos2dBufferMemory);

	VkBuffer adamMoments;
	VkDeviceMemory adamMomentsMemory;
	createBuffer(sizeof(AdamState) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				adamMoments, adamMomentsMemory);

	VkBuffer flagBuffer;
	VkDeviceMemory flagBufferMemory;
	createBuffer(sizeof(uint32_t) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				flagBuffer, flagBufferMemory);

	VkBuffer activeGaussianCountBuffer; // for density control, to prevent invalid access during density control
	VkDeviceMemory activeGaussianCountBufferMemory;
	createBuffer(sizeof(uint32_t),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				activeGaussianCountBuffer, activeGaussianCountBufferMemory);

	createBuffer(sizeof(uint32_t) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				contributionCountBuffer, contributionCountBufferMemory);

	// --- Stream Compaction Buffers ---
	VkBuffer scanBuffer; // Stores the exclusive prefix sum (indices)
	VkDeviceMemory scanBufferMemory;
	createBuffer(sizeof(uint32_t) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				scanBuffer, scanBufferMemory);

	uint32_t sum1Size = (maxGaussians + 511) / 512;
	VkBuffer blockSumBuffer1;
	VkDeviceMemory blockSumBuffer1Memory;
	createBuffer(sizeof(uint32_t) * sum1Size,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				blockSumBuffer1, blockSumBuffer1Memory);

	uint32_t sum2Size = (sum1Size + 511) / 512;
	VkBuffer blockSumBuffer2;
	VkDeviceMemory blockSumBuffer2Memory;
	createBuffer(sizeof(uint32_t) * sum2Size,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				blockSumBuffer2, blockSumBuffer2Memory);

	VkBuffer tmpGaussianBuffer;
	VkDeviceMemory tmpGaussianBufferMemory;
	createBuffer(sizeof(Gaussian3D) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				tmpGaussianBuffer, tmpGaussianBufferMemory);

	VkBuffer tmpAdamBuffer;
	VkDeviceMemory tmpAdamBufferMemory;
	createBuffer(sizeof(AdamState) * maxGaussians,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				tmpAdamBuffer, tmpAdamBufferMemory);

	VkBuffer dummyBuffer; // For scan L3 output
	VkDeviceMemory dummyBufferMemory;
	createBuffer(sizeof(uint32_t) * 512,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				dummyBuffer, dummyBufferMemory);

	createImage(render_width, render_height, VK_FORMAT_R32_UINT,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				outContribution, outContributionMemory);
	createImage(render_width, render_height, VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				gtImage, gtImageMemory);
	createImage(render_width, render_height, VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				curSSIMStats, curSSIMStatsMemory);
	createImage(render_width, render_height, VK_FORMAT_R32G32_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				gtSSIMStats, gtSSIMStatsMemory);
	createImage(render_width, render_height, VK_FORMAT_R32G32B32A32_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				tempSSIMStats, tempSSIMStatsMemory);
	createImage(render_width, render_height, VK_FORMAT_R32_SFLOAT,
				VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				tempSSIMStats2, tempSSIMStatsMemory2);

	VkImageView outContributionImageView;
	VkImageView gtImageView;
	VkImageView curSSIMStatsImageView;
	VkImageView gtSSIMStatsImageView;
	VkImageView tempSSIMStatsImageView;
	VkImageView tempSSIMStatsImageView2;

	createImageView(VK_FORMAT_R32_UINT, outContribution, outContributionImageView);
	createImageView(VK_FORMAT_R8G8B8A8_UNORM, gtImage, gtImageView);
	createImageView(VK_FORMAT_R32G32B32A32_SFLOAT, curSSIMStats, curSSIMStatsImageView);
	createImageView(VK_FORMAT_R32G32_SFLOAT, gtSSIMStats, gtSSIMStatsImageView);
	createImageView(VK_FORMAT_R32G32B32A32_SFLOAT, tempSSIMStats, tempSSIMStatsImageView);
	createImageView(VK_FORMAT_R32_SFLOAT, tempSSIMStats2, tempSSIMStatsImageView2);

	transitionImageLayout(outContribution, VK_FORMAT_R32_UINT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(gtImage, VK_FORMAT_R8G8B8A8_UNORM,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(curSSIMStats, VK_FORMAT_R32G32B32A32_SFLOAT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(gtSSIMStats, VK_FORMAT_R32G32_SFLOAT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(tempSSIMStats, VK_FORMAT_R32G32B32A32_SFLOAT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(tempSSIMStats2, VK_FORMAT_R32_SFLOAT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	transitionImageLayout(offscreenImages[0], VK_FORMAT_R8G8B8_UNORM,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

	// initialize buffers
	{
		VkCommandBuffer clearCmd = beginSingleTimeCommands();
		vkCmdFillBuffer(clearCmd, adamMoments, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(clearCmd, gradPos2dBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(clearCmd, activeGaussianCountBuffer, 0, VK_WHOLE_SIZE, totalGaussians);
		endSingleTimeCommands(clearCmd);
	}

	std::array<VkDescriptorSetLayoutBinding, 15> trainBindings{};

	trainBindings[0] = { // camera buffer
						.binding = 0,
						.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[1] = { // outimage
						.binding = 1,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[2] = { // gtImage
						.binding = 2,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[3] = { // outContribution
						.binding = 3,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[4] = { // cur ssim stats
						.binding = 4,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[5] = { // gt ssim stats
						.binding = 5,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[6] = { // temp ssim stats
						.binding = 6,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[7] = { // temp ssim stats
						.binding = 7,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[8] = { // gaussianGrad
						.binding = 8,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[9] = { // gradPos2d
						.binding = 9,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[10] = { // adamMoments
						.binding = 10,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[11] = { // flag buffer
						.binding = 11,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[12] = { // active gaussian counter buffer
						.binding = 12,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[13] = { // scan buffer / indices
						.binding = 13,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};
	trainBindings[14] = { // contribution count buffer
						.binding = 14,
						.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						.descriptorCount = 1,
						.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
						.pImmutableSamplers = nullptr};

	VkDescriptorSetLayoutCreateInfo trainSetInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.bindingCount = static_cast<uint32_t>(trainBindings.size()),
		.pBindings = trainBindings.data()};
	VkDescriptorSetLayout trainSetLayout;
	vkCreateDescriptorSetLayout(device, &trainSetInfo, nullptr, &trainSetLayout);

	VkDescriptorSet trainDescriptorSet;
	VkDescriptorPool trainPool;

	std::array<VkDescriptorPoolSize, 3> poolSizes;
	poolSizes[0] = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1};
	poolSizes[1] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
					.descriptorCount = 7};
	poolSizes[2] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 11};

  VkDescriptorPoolCreateInfo poolInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.maxSets = 1,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()};
	vkCreateDescriptorPool(device, &poolInfo, nullptr, &trainPool);

	VkDescriptorSetAllocateInfo allocInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = nullptr,
		.descriptorPool = trainPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &trainSetLayout};
	vkAllocateDescriptorSets(device, &allocInfo, &trainDescriptorSet);

	VkDescriptorBufferInfo camInfo{
		.buffer = cameraBuffers[0], .offset = 0, .range = sizeof(CameraUBO)};
	VkDescriptorImageInfo outImageInfo{.sampler = VK_NULL_HANDLE,
										.imageView = offscreenImageViews[0],
										.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo gtImageInfo{.sampler = VK_NULL_HANDLE,
										.imageView = gtImageView,
										.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo outContributionInfo{
		.sampler = VK_NULL_HANDLE,
		.imageView = outContributionImageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo curSSIMStatsInfo{
		.sampler = VK_NULL_HANDLE,
		.imageView = curSSIMStatsImageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo gtSSIMStatsInfo{
		.sampler = VK_NULL_HANDLE,
		.imageView = gtSSIMStatsImageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo tempSSIMStatsInfo{
		.sampler = VK_NULL_HANDLE,
		.imageView = tempSSIMStatsImageView,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo tempSSIMStatsInfo2{
		.sampler = VK_NULL_HANDLE,
		.imageView = tempSSIMStatsImageView2,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorBufferInfo gaussianGradInfo{
		.buffer = gaussianGradBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo gradPos2dInfo{
		.buffer = gradPos2dBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo adamMomentInfo{
		.buffer = adamMoments, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo flagBufferInfo{
		.buffer = flagBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo activeGaussianCountBufferInfo{
		.buffer = activeGaussianCountBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo scanBufferInfo{
		.buffer = scanBuffer, .offset = 0, .range = VK_WHOLE_SIZE};
	VkDescriptorBufferInfo contributionCountInfo{
		.buffer = contributionCountBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

	std::array<VkWriteDescriptorSet, 15> writes{};
	writes[0] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &camInfo};
	writes[1] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &outImageInfo};
	writes[2] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 2,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &gtImageInfo};
	writes[3] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 3,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &outContributionInfo};
	writes[4] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 4,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &curSSIMStatsInfo};
	writes[5] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 5,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &gtSSIMStatsInfo};
	writes[6] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 6,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &tempSSIMStatsInfo};
	writes[7] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 7,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &tempSSIMStatsInfo2};
	writes[8] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 8,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &gaussianGradInfo};
	writes[9] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 9,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &gradPos2dInfo};
	writes[10] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 10,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &adamMomentInfo};
	writes[11] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 11,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &flagBufferInfo};
	writes[12] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 12,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &activeGaussianCountBufferInfo};
	writes[13] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 13,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &scanBufferInfo};
	writes[14] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = trainDescriptorSet,
				.dstBinding = 14,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &contributionCountInfo};

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	std::array<VkDescriptorSetLayout, 2> trainSetLayouts = {globalDescriptorSetLayout, trainSetLayout};
	VkDescriptorSet bindTrainDescriptorSets[] = {globalDescriptorSets, trainDescriptorSet};

	VkPipeline rasterTrainComputePipeline;
	VkPipelineLayout rasterTrainComputePipelineLayout;
	VkPushConstantRange rasterPushRange{};
	rasterPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	rasterPushRange.offset = 0;
	rasterPushRange.size = sizeof(RasterPush);
	createComputePipeline(rasterTrainComputePipeline, rasterTrainComputePipelineLayout,
						"../shader/spv/raster_train.spv", 1, &rasterPushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());

	VkPushConstantRange convPushRange = rasterPushRange;
	convPushRange.size = sizeof(int);
	VkPipeline convComputePipeline;
	VkPipelineLayout convComputePipelineLayout;
	createComputePipeline(convComputePipeline, convComputePipelineLayout,
						"../shader/spv/convolute.spv", 1, &convPushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());

	VkPushConstantRange backPushRange = rasterPushRange;
	backPushRange.size = sizeof(BackwardPush);
	VkPipeline backwardComputePipeline;
	VkPipelineLayout backwardComputePipelineLayout;
	createComputePipeline(backwardComputePipeline, backwardComputePipelineLayout,
						"../shader/spv/backward.spv", 1, &backPushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());

	VkPushConstantRange adamPushRange = backPushRange;
	adamPushRange.size = sizeof(AdamPush);
	VkPipeline adamComputePipeline;
	VkPipelineLayout adamComputePipelineLayout;
	createComputePipeline(adamComputePipeline, adamComputePipelineLayout,
						"../shader/spv/adam.spv", 1, &adamPushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());

	// --- Stream Compaction Pipelines ---
	VkDescriptorSetLayout scanSetLayout;
	{
		std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
		bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo info{
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			nullptr, 0, 2, bindings.data()};
		vkCreateDescriptorSetLayout(device, &info, nullptr, &scanSetLayout);
	}
	VkDescriptorSetLayout compactSpecificSetLayout;
	{
		std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
		bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // dst gaussians
		bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}; // dst adam
		VkDescriptorSetLayoutCreateInfo info{
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			nullptr, 0, 2, bindings.data()};
		vkCreateDescriptorSetLayout(device, &info, nullptr, &compactSpecificSetLayout);
	}

	VkDescriptorPool scanPool;
	{
		std::array<VkDescriptorPoolSize, 1> sizes = {{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}}};
		VkDescriptorPoolCreateInfo info{
			VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			nullptr, 0, 4, 1, sizes.data()};
		vkCreateDescriptorPool(device, &info, nullptr, &scanPool);
	}

	VkDescriptorSet scanSetL1, scanSetL2, scanSetL3, compactSet;
	{
		VkDescriptorSetAllocateInfo alloc{
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			nullptr, scanPool, 1, &scanSetLayout};
		vkAllocateDescriptorSets(device, &alloc, &scanSetL1);
		vkAllocateDescriptorSets(device, &alloc, &scanSetL2);
		vkAllocateDescriptorSets(device, &alloc, &scanSetL3);
		alloc.pSetLayouts = &compactSpecificSetLayout;
		vkAllocateDescriptorSets(device, &alloc, &compactSet);
	}

	// Update Scan Sets
	{
		auto updateScanSet = [&](VkDescriptorSet set, VkBuffer b0, VkBuffer b1) {
			VkDescriptorBufferInfo i0{b0, 0, VK_WHOLE_SIZE};
			VkDescriptorBufferInfo i1{b1, 0, VK_WHOLE_SIZE};
			std::array<VkWriteDescriptorSet, 2> w{};
			w[0] = {
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &i0, nullptr};
			w[1] = {
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &i1, nullptr};
			vkUpdateDescriptorSets(device, 2, w.data(), 0, nullptr);
		};
		updateScanSet(scanSetL1, scanBuffer, blockSumBuffer1);
		updateScanSet(scanSetL2, blockSumBuffer1, blockSumBuffer2);
		updateScanSet(scanSetL3, blockSumBuffer2, dummyBuffer);

		VkDescriptorBufferInfo i0{tmpGaussianBuffer, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo i1{tmpAdamBuffer, 0, VK_WHOLE_SIZE};
		std::array<VkWriteDescriptorSet, 2> w{};
		w[0] = {
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compactSet, 0, 0, 1,
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &i0, nullptr};
		w[1] = {
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compactSet, 1, 0, 1,
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &i1, nullptr};
		vkUpdateDescriptorSets(device, 2, w.data(), 0, nullptr);
	}

	VkPipeline markPrunePipeline;
	VkPipelineLayout markPrunePipelineLayout;
	VkPushConstantRange prunePushRange{};
	prunePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	prunePushRange.offset = 0;
	prunePushRange.size = sizeof(float); // sceneExtent
	createComputePipeline(markPrunePipeline, markPrunePipelineLayout,
						"../shader/spv/mark_prune.spv", 1, &prunePushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());


	VkPipeline scanLocalPipeline;
	VkPipelineLayout scanLocalPipelineLayout;
	createComputePipeline(scanLocalPipeline, scanLocalPipelineLayout,
						"../shader/spv/scan_local.spv", 0, nullptr, 1, &scanSetLayout);

	VkPipeline scanAddPipeline;
	VkPipelineLayout scanAddPipelineLayout;
	createComputePipeline(scanAddPipeline, scanAddPipelineLayout,
						"../shader/spv/scan_add.spv", 0, nullptr, 1, &scanSetLayout);

	VkPipeline compactPipeline;
	VkPipelineLayout compactPipelineLayout;
	std::array<VkDescriptorSetLayout, 3> compactLayouts = {
		globalDescriptorSetLayout, trainSetLayout, compactSpecificSetLayout};
	createComputePipeline(compactPipeline, compactPipelineLayout,
						"../shader/spv/compact.spv", 0, nullptr, 3, compactLayouts.data());

	VkPipeline densityControlPipeline;
	VkPipelineLayout densityControlPipelineLayout;
	VkPushConstantRange densityControlPushRange = adamPushRange;
	densityControlPushRange.size = sizeof(DensityControlPush);
	createComputePipeline(densityControlPipeline, densityControlPipelineLayout,
						"../shader/spv/density_control.spv",
						1, &densityControlPushRange,
						static_cast<uint32_t>(trainSetLayouts.size()), trainSetLayouts.data());

	VkBuffer imageStagingBuffer;
	VkDeviceMemory imageStagingBufferMemory;
	createBuffer(render_width * render_height * 4,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				imageStagingBuffer, imageStagingBufferMemory);
	void *data;
	vkMapMemory(device, imageStagingBufferMemory, 0, render_width * render_height * 4, 0, &data);

	float sceneExt = calculateSceneExtent(images);

	uint32_t steps = 0;
	const int total_iterations = iterations; // Save original for LR decay
	int frameIdx = 0; // Use symmetric 1-frame rendering for training instead of
	// MAX_FRAME_IN_FLIGHT to prevent data races
	while (iterations--) {
		++steps;
		if (steps % 100 == 0) {
			printf("Steps : %d\n", steps);
		}

		VkCommandBuffer cmdbuf = computeCommandBuffers[frameIdx];
		VkCommandBuffer cmdbuf2 = computeCommandBuffers2[frameIdx];

		// --- Synchronization & Resets ---
		vkWaitForFences(device, 1, &computeInFlightFences[frameIdx], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeInFlightFences[frameIdx]);

		vkResetCommandBuffer(cmdbuf, 0);
		vkResetCommandBuffer(cmdbuf2, 0);

		// --- Resource Updates (Move after Fence Wait to avoid Data Race) ---
		float current_lr = lr * pow(0.01f, (float)steps / 30000.0f);
		// Pick a random image for training
		int imgIdx = rand() % images.size();
		const Image &img = images[imgIdx];

		// Find corresponding camera for intrinsics
		const Camera *pCam = nullptr;
		for (const auto &c : cameras) {
			if (c.id == img.camera_id) {
				pCam = &c;
				break;
			}
		}
		setCameraFromColmap(img, pCam);

		// Load ground truth image if not in cache
		if (gtImageCache[imgIdx].empty()) {
			std::string imagePath = "../dense/images/" + img.name;
			int texWidth, texHeight, texChannels;
			stbi_uc *pixels = stbi_load(imagePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			if (!pixels) {
				throw std::runtime_error("Failed to load ground truth image: " + imagePath);
			}

			gtImageCache[imgIdx].resize(render_width * render_height * 4);
			if (texWidth != render_width || texHeight != render_height) {
				stbir_resize_uint8_linear(pixels, texWidth, texHeight, 0, gtImageCache[imgIdx].data(), render_width, render_height, 0, STBIR_RGBA);
			} else {
				memcpy(gtImageCache[imgIdx].data(), pixels, render_width * render_height * 4);
			}
			stbi_image_free(pixels);
    	}

		// Upload to staging buffer
		memcpy(data, gtImageCache[imgIdx].data(), render_width * render_height * 4);

		VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		// --- Pass A (Projection & Argpass) ---
		vkBeginCommandBuffer(cmdbuf, &beginInfo);

		// Transition gtImage to TRANSFER_DST_OPTIMAL
		VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = gtImage;
		barrier.subresourceRange = subresourceRange;

		vkCmdPipelineBarrier(cmdbuf, 
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_TRANSFER_BIT, 
							0, 0, nullptr, 0, nullptr, 1, &barrier);

		VkBufferImageCopy region{};
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		region.imageExtent = {render_width, render_height, 1};
		vkCmdCopyBufferToImage(cmdbuf, imageStagingBuffer, gtImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// Transition gtImage back to GENERAL
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmdbuf,
							VK_PIPELINE_STAGE_TRANSFER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 0, nullptr, 0, nullptr, 1, &barrier);

		vkCmdFillBuffer(cmdbuf, gaussianGradBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmdbuf, counterBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmdbuf, tileRangeBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmdbuf, projectedGaussianBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmdbuf, keyBuffer, 0, VK_WHOLE_SIZE, 0);
		vkCmdFillBuffer(cmdbuf, valueBuffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFF);

		// Clear outContribution to sentinel value (0xFFFFFFFF)
		VkClearColorValue clearValue;
		clearValue.uint32[0] = 0xFFFFFFFFu;
		clearValue.uint32[1] = 0xFFFFFFFFu;
		clearValue.uint32[2] = 0xFFFFFFFFu;
		clearValue.uint32[3] = 0xFFFFFFFFu;

		VkImageSubresourceRange clearRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		vkCmdClearColorImage(cmdbuf, outContribution, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

		VkMemoryBarrier initBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		initBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		initBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmdbuf,
							VK_PIPELINE_STAGE_TRANSFER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
							0, 1, &initBarrier, 0, nullptr, 0, nullptr);

		vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpass2Pipeline);
		vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
								argpass2PipelineLayout, 0, 1, &globalDescriptorSets,
								0, nullptr);
		vkCmdDispatch(cmdbuf, 1, 1, 1);

		VkMemoryBarrier postArgpassBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		postArgpassBarrier.srcAccessMask =VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		postArgpassBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		vkCmdPipelineBarrier(cmdbuf,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
							0, 1, &postArgpassBarrier, 0, nullptr, 0, nullptr);

		vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, projComputePipeline);
		ProjPush projPush{KVCapacity, (uint32_t)n_tiles_col, (uint32_t)n_tiles_row};
		vkCmdPushConstants(cmdbuf, projComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProjPush), &projPush);
		VkDescriptorSet bindDescriptorSets[] = {globalDescriptorSets, localDescriptorSets[frameIdx]};
		vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
								projComputePipelineLayout, 0, 2, bindDescriptorSets,
								0, nullptr);
		vkCmdDispatchIndirect(cmdbuf, indirectArgsBuffer, 0);

		VkMemoryBarrier midPassABarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		midPassABarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		midPassABarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmdbuf,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 1, &midPassABarrier, 0, nullptr, 0, nullptr);

		vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, argpassComputePipeline);
		vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
								argpassComputePipelineLayout, 0, 2,
								bindDescriptorSets, 0, nullptr);
		vkCmdPushConstants(cmdbuf, argpassComputePipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
							&KVCapacity);
		vkCmdDispatch(cmdbuf, 1, 1, 1);

		// Visibility Barrier: Ensure Pass A writes are visible to Pass B
		VkMemoryBarrier passAFinalBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		passAFinalBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		passAFinalBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
		vkCmdPipelineBarrier(cmdbuf,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
							0, 1, &passAFinalBarrier, 0, nullptr, 0, nullptr);

		vkEndCommandBuffer(cmdbuf);

		VkSubmitInfo submitInfoA{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submitInfoA.commandBufferCount = 1;
		submitInfoA.pCommandBuffers = &cmdbuf;
		submitInfoA.signalSemaphoreCount = 1;
		submitInfoA.pSignalSemaphores = &projectionFinishedSemaphores[frameIdx];
		vkQueueSubmit(computeQueue, 1, &submitInfoA, VK_NULL_HANDLE);

		// --- Pass B (Sort, Range, Raster) ---
		vkBeginCommandBuffer(cmdbuf2, &beginInfo);
		vrdxCmdSortKeyValueIndirect(cmdbuf2, sorter, KVCapacity, counterBuffer, 0,
									keyBuffer, 0, valueBuffer, 0, pingpongBuffer, 0,
									VK_NULL_HANDLE, 0);

		VkMemoryBarrier postSortBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		postSortBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
		postSortBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
							0, 1, &postSortBarrier, 0, nullptr, 0, nullptr);

		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								rangeComputePipelineLayout, 0, 1,
								&globalDescriptorSets, 0, nullptr);
		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rangeComputePipeline);
		vkCmdPushConstants(cmdbuf2, rangeComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
						&num_tiles);
		vkCmdDispatchIndirect(cmdbuf2, indirectArgsBuffer, 0);

		VkImageMemoryBarrier imageToGeneralBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		imageToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageToGeneralBarrier.image = offscreenImages[frameIdx];
		imageToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageToGeneralBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		imageToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageToGeneralBarrier.subresourceRange = subresourceRange;

		VkImageMemoryBarrier contribToGeneralBarrier = imageToGeneralBarrier;
		contribToGeneralBarrier.image = outContribution;

		std::array<VkImageMemoryBarrier, 2> imageBarriers = {imageToGeneralBarrier, contribToGeneralBarrier};

		VkMemoryBarrier postRangeBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		postRangeBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
		postRangeBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 1, &postRangeBarrier, 0, nullptr, (uint32_t)imageBarriers.size(), imageBarriers.data());

		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, rasterTrainComputePipeline);
		// Generate random solid background color
		// float bgR = static_cast<float>(rand()) / RAND_MAX;
		// float bgG = static_cast<float>(rand()) / RAND_MAX;
		// float bgB = static_cast<float>(rand()) / RAND_MAX;
		float bgR = 0;
		float bgG = 0;
		float bgB = 0;
		RasterPush rasterPush{KVCapacity, bgR, bgG, bgB, (uint32_t)n_tiles_col, (uint32_t)n_tiles_row};
		vkCmdPushConstants(cmdbuf2, rasterTrainComputePipelineLayout,
						VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RasterPush),
						&rasterPush);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								rasterTrainComputePipelineLayout, 0, 2,
								bindTrainDescriptorSets, 0, nullptr);
		vkCmdDispatch(cmdbuf2, (render_width + 15) / 16, (render_height + 15) / 16, 1);

		// backward 
		VkMemoryBarrier preConvBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		preConvBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		preConvBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
							&preConvBarrier, 0, nullptr, 0, nullptr);

		int vertical = 0;
		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, convComputePipeline);
		vkCmdPushConstants(cmdbuf2, convComputePipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &vertical);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								convComputePipelineLayout, 0, 2,
								bindTrainDescriptorSets, 0, nullptr);
		vkCmdDispatch(cmdbuf2, (render_width + 15) / 16, (render_height + 15) / 16, 1);

		VkMemoryBarrier midConvBarrier = preConvBarrier;
		vkCmdPipelineBarrier(cmdbuf2, 
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 1, &midConvBarrier, 0, nullptr, 0, nullptr);

		vertical = 1;
		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, convComputePipeline);
		vkCmdPushConstants(cmdbuf2, convComputePipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &vertical);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								convComputePipelineLayout, 0, 2,
								bindTrainDescriptorSets, 0, nullptr);
		vkCmdDispatch(cmdbuf2, (render_width + 15) / 16, (render_height + 15) / 16, 1);

		VkMemoryBarrier postConvBarrier = preConvBarrier;
		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 1, &postConvBarrier, 0, nullptr, 0, nullptr);

		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, backwardComputePipeline);
		BackwardPush backPush{lambda, bgR, bgG, bgB, KVCapacity, (uint32_t)n_tiles_col, (uint32_t)n_tiles_row};
		vkCmdPushConstants(cmdbuf2, backwardComputePipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BackwardPush),
							&backPush);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								backwardComputePipelineLayout, 0, 2,
								bindTrainDescriptorSets, 0, nullptr);
		vkCmdDispatch(cmdbuf2, (render_width + 15) / 16, (render_height + 15) / 16, 1);

		// 저장한 그래디언트를 통한 역전파 및 밀도 조정
		VkMemoryBarrier preAdamBarrier = preConvBarrier;
		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							0, 1, &preAdamBarrier, 0, nullptr, 0, nullptr);

		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, argpass2Pipeline);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								argpass2PipelineLayout, 0, 1, &globalDescriptorSets,
								0, nullptr);
		vkCmdDispatch(cmdbuf2, 1, 1, 1);

		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
							0, 1, &postArgpassBarrier, 0, nullptr, 0, nullptr);

		// WP : MUST CHANGE TO INDIRECT DISPATCH since totalGaussian will be
		// changed!! (Use gaussianCountBuffer)
		vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, adamComputePipeline);
		AdamPush adamPush{
			current_lr, // pos_lr (exponentially decayed from default 0.00016)
			0.001f,     // rot_lr
			0.005f,     // scale_lr
			0.05f,      // opacity_lr
			0.0025f,    // color_lr
			beta1, 
			beta2, 
			eps, 
			steps
		};
		vkCmdPushConstants(cmdbuf2, adamComputePipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdamPush),
							&adamPush);
		vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
								adamComputePipelineLayout, 0, 2,
								bindTrainDescriptorSets, 0, nullptr);
		vkCmdDispatchIndirect(cmdbuf2, indirectArgsBuffer, 0);

		// Every 100 steps, after densification, or at the start, clear accumulation
		if (steps % 100 == 1) {
			vkCmdFillBuffer(cmdbuf2, gradPos2dBuffer, 0, VK_WHOLE_SIZE, 0);
			vkCmdFillBuffer(cmdbuf2, contributionCountBuffer, 0, VK_WHOLE_SIZE, 0);
		}

		VkMemoryBarrier preCompactBarrier = preConvBarrier;
		preCompactBarrier.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		vkCmdPipelineBarrier(cmdbuf2,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
							0, 1, &preCompactBarrier, 0, nullptr, 0, nullptr);

		// --- Density Control and Compaction ---
		if (steps % 100 == 0) {
			// 1. Density Control (Perform BEFORE compaction to use correct indices)
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, densityControlPipeline);
			DensityControlPush densityPush{
				.sceneExtent = sceneExt, .maxGaussians = maxGaussians, .step = steps};
			vkCmdPushConstants(cmdbuf2, densityControlPipelineLayout,
								VK_SHADER_STAGE_COMPUTE_BIT, 0,
								sizeof(DensityControlPush), &densityPush);
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									densityControlPipelineLayout, 0, 2,
									bindTrainDescriptorSets, 0, nullptr);

			VkBufferCopy bufferRegionInit{0, 0, sizeof(uint32_t)};
			vkCmdCopyBuffer(cmdbuf2, gaussianCountBuffer, activeGaussianCountBuffer, 1, &bufferRegionInit);

			VkMemoryBarrier preDensityControlBarrier{
				VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_TRANSFER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
								0, 1, &preDensityControlBarrier, 0, nullptr, 0, nullptr);

			vkCmdDispatchIndirect(cmdbuf2, indirectArgsBuffer, 0);

			VkMemoryBarrier postDensityBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
			postDensityBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			postDensityBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								0, 1, &postDensityBarrier, 0, nullptr, 0, nullptr);

			// 2. Mark Prune
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, markPrunePipeline);

			// Update activeGaussianCountBuffer from gaussianCountBuffer after
			// densification
			vkCmdCopyBuffer(cmdbuf2, gaussianCountBuffer, activeGaussianCountBuffer, 1, &bufferRegionInit);

			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_TRANSFER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
								0, 1, &preDensityControlBarrier, 0, nullptr, 0, nullptr);
			
			vkCmdPushConstants(cmdbuf2, markPrunePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &sceneExt);

			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									markPrunePipelineLayout, 0, 2,
									bindTrainDescriptorSets, 0, nullptr);


			// Current maxGaussians for pruning (not active_count as we might have added new ones)
			vkCmdDispatch(cmdbuf2, (maxGaussians + 255) / 256, 1, 1);

			VkMemoryBarrier markBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
										VK_ACCESS_SHADER_WRITE_BIT,
										VK_ACCESS_TRANSFER_READ_BIT};
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_TRANSFER_BIT, 
								0, 1, &markBarrier, 0, nullptr, 0, nullptr);

			// 3. Initialize Scan Buffer
			VkBufferCopy copyRegion{0, 0, sizeof(uint32_t) * maxGaussians};
			vkCmdCopyBuffer(cmdbuf2, flagBuffer, scanBuffer, 1, &copyRegion);

			VkMemoryBarrier copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
										VK_ACCESS_TRANSFER_WRITE_BIT,
										VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_TRANSFER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								0, 1, &copyBarrier, 0, nullptr, 0, nullptr);

			// 4. Multi-level Scan
			// Level 1
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, scanLocalPipeline);
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									scanLocalPipelineLayout, 0, 1, &scanSetL1, 0,
									nullptr);
			vkCmdDispatch(cmdbuf2, (maxGaussians + 511) / 512, 1, 1);

			VkMemoryBarrier scanL1Barrier{
				VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								0, 1, &scanL1Barrier, 0, nullptr, 0, nullptr);

			// Level 2
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									scanLocalPipelineLayout, 0, 1, &scanSetL2, 0,
									nullptr);
			vkCmdDispatch(cmdbuf2, (sum1Size + 511) / 512, 1, 1);

			VkMemoryBarrier scanL2Barrier = scanL1Barrier;
			vkCmdPipelineBarrier(cmdbuf2,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								0, 1, &scanL2Barrier, 0, nullptr, 0, nullptr);

			// Level 3
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									scanLocalPipelineLayout, 0, 1, &scanSetL3, 0, nullptr);
			vkCmdDispatch(cmdbuf2, 1, 1, 1);

			VkMemoryBarrier scanL3Barrier = scanL1Barrier;
			vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
								&scanL3Barrier, 0, nullptr, 0, nullptr);

			// Level 2 Add
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, scanAddPipeline);
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									scanAddPipelineLayout, 0, 1, &scanSetL2, 0,
									nullptr);
			vkCmdDispatch(cmdbuf2, (sum1Size + 511) / 512, 1, 1);

			VkMemoryBarrier addL2Barrier = scanL1Barrier;
			vkCmdPipelineBarrier(cmdbuf2, 
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								0, 1, &addL2Barrier, 0, nullptr, 0, nullptr);

			// Level 1 Add
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									scanAddPipelineLayout, 0, 1, &scanSetL1, 0,
									nullptr);
			vkCmdDispatch(cmdbuf2, (maxGaussians + 511) / 512, 1, 1);

			VkMemoryBarrier addL1Barrier = scanL1Barrier;
			vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
								&addL1Barrier, 0, nullptr, 0, nullptr);

			// 5. Compact
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, compactPipeline);
			vkCmdFillBuffer(cmdbuf2, gaussianCountBuffer, 0, sizeof(uint32_t), 0); // Reset counter for atomicMax
			VkMemoryBarrier resetBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
										VK_ACCESS_TRANSFER_WRITE_BIT,
										VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
			vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_TRANSFER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
								&resetBarrier, 0, nullptr, 0, nullptr);

			VkDescriptorSet compactBindSets[] = {globalDescriptorSets, trainDescriptorSet, compactSet};
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									compactPipelineLayout, 0, 3, compactBindSets, 0,
									nullptr);
			vkCmdDispatch(cmdbuf2, (maxGaussians + 255) / 256, 1, 1);

			VkMemoryBarrier compactBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
											VK_ACCESS_SHADER_WRITE_BIT,
											VK_ACCESS_TRANSFER_READ_BIT};
			vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
								&compactBarrier, 0, nullptr, 0, nullptr);

			// 6. Transfer Back
			VkBufferCopy fullCopy{0, 0, sizeof(Gaussian3D) * maxGaussians};
			vkCmdCopyBuffer(cmdbuf2, tmpGaussianBuffer, gaussianBuffer, 1, &fullCopy);
			VkBufferCopy adamCopy{0, 0, sizeof(AdamState) * maxGaussians};
			vkCmdCopyBuffer(cmdbuf2, tmpAdamBuffer, adamMoments, 1, &adamCopy);

			VkMemoryBarrier finalBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
										VK_ACCESS_TRANSFER_WRITE_BIT,
										VK_ACCESS_SHADER_READ_BIT};
			vkCmdPipelineBarrier(cmdbuf2, VK_PIPELINE_STAGE_TRANSFER_BIT,
								VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
								&finalBarrier, 0, nullptr, 0, nullptr);

			// 7. Final Argpass to update indirectArgsBuffer for normal training
			vkCmdBindPipeline(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE, argpass2Pipeline);
			vkCmdBindDescriptorSets(cmdbuf2, VK_PIPELINE_BIND_POINT_COMPUTE,
									argpass2PipelineLayout, 0, 1,
									&globalDescriptorSets, 0, nullptr);
			vkCmdDispatch(cmdbuf2, 1, 1, 1);
		}

    	vkEndCommandBuffer(cmdbuf2);

		VkSubmitInfo submitInfoB{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		VkPipelineStageFlags computeWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
		submitInfoB.commandBufferCount = 1;
		submitInfoB.pCommandBuffers = &cmdbuf2;
		submitInfoB.waitSemaphoreCount = 1;
		submitInfoB.pWaitSemaphores = &projectionFinishedSemaphores[frameIdx];
		submitInfoB.pWaitDstStageMask = computeWaitStages;
		submitInfoB.signalSemaphoreCount = 0;
		submitInfoB.pSignalSemaphores = nullptr;

		vkQueueSubmit(computeQueue, 1, &submitInfoB, computeInFlightFences[frameIdx]);
	}

  	vkDeviceWaitIdle(device);

	// Sync Gaussian count from GPU to CPU
	{
		VkBuffer countStagingBuffer;
		VkDeviceMemory countStagingBufferMemory;
		createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					countStagingBuffer, countStagingBufferMemory);

		copyBuffer(activeGaussianCountBuffer, countStagingBuffer, sizeof(uint32_t));

		void *pCountData;
		vkMapMemory(device, countStagingBufferMemory, 0, sizeof(uint32_t), 0, &pCountData);
		memcpy(&totalGaussians, pCountData, sizeof(uint32_t));
		vkUnmapMemory(device, countStagingBufferMemory);

		vkDestroyBuffer(device, countStagingBuffer, nullptr);
		vkFreeMemory(device, countStagingBufferMemory, nullptr);
	}

	printf("Final total Gaussians: %u\n", totalGaussians);
	if (totalGaussians > maxGaussians) {
		printf("Warning: totalGaussians (%u) exceeded maxGaussians (%u). Clamping to max.\n", totalGaussians, maxGaussians);
		totalGaussians = maxGaussians;
	}

	// export gaussians count, params
	VkDeviceSize exportSize = sizeof(Gaussian3D) * totalGaussians;
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(exportSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

	copyBuffer(gaussianBuffer, stagingBuffer, exportSize);

	void *pExportData;
	vkMapMemory(device, stagingBufferMemory, 0, exportSize, 0, &pExportData);
	std::vector<Gaussian3D> exportedGaussians(totalGaussians);
	memcpy(exportedGaussians.data(), pExportData, exportSize);
	vkUnmapMemory(device, stagingBufferMemory);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);

	Core::exportGaussians("trained_gaussians.bin", exportedGaussians, totalGaussians);

	vkUnmapMemory(device, imageStagingBufferMemory);
	vkDestroyBuffer(device, imageStagingBuffer, nullptr);
	vkFreeMemory(device, imageStagingBufferMemory, nullptr);

	vkDestroyPipeline(device, rasterTrainComputePipeline, nullptr);
	vkDestroyPipelineLayout(device, rasterTrainComputePipelineLayout, nullptr);
	vkDestroyPipeline(device, convComputePipeline, nullptr);
	vkDestroyPipelineLayout(device, convComputePipelineLayout, nullptr);
	vkDestroyPipeline(device, backwardComputePipeline, nullptr);
	vkDestroyPipelineLayout(device, backwardComputePipelineLayout, nullptr);
	vkDestroyPipeline(device, adamComputePipeline, nullptr);
	vkDestroyPipelineLayout(device, adamComputePipelineLayout, nullptr);
	vkDestroyDescriptorPool(device, trainPool, nullptr);
	vkDestroyDescriptorSetLayout(device, trainSetLayout, nullptr);

	vkDestroyImageView(device, gtSSIMStatsImageView, nullptr);
	vkDestroyImage(device, gtSSIMStats, nullptr);
	vkFreeMemory(device, gtSSIMStatsMemory, nullptr);

	vkDestroyBuffer(device, gaussianGradBuffer, nullptr);
	vkFreeMemory(device, gaussianGradBufferMemory, nullptr);
	vkDestroyBuffer(device, gradPos2dBuffer, nullptr);
	vkFreeMemory(device, gradPos2dBufferMemory, nullptr);
	vkDestroyBuffer(device, adamMoments, nullptr);
	vkFreeMemory(device, adamMomentsMemory, nullptr);
	vkDestroyBuffer(device, contributionCountBuffer, nullptr);
	vkFreeMemory(device, contributionCountBufferMemory, nullptr);
	vkDestroyBuffer(device, flagBuffer, nullptr);
	vkFreeMemory(device, flagBufferMemory, nullptr);
	vkDestroyBuffer(device, activeGaussianCountBuffer, nullptr);
	vkFreeMemory(device, activeGaussianCountBufferMemory, nullptr);

	vkDestroyBuffer(device, scanBuffer, nullptr);
	vkFreeMemory(device, scanBufferMemory, nullptr);
	vkDestroyBuffer(device, blockSumBuffer1, nullptr);
	vkFreeMemory(device, blockSumBuffer1Memory, nullptr);
	vkDestroyBuffer(device, blockSumBuffer2, nullptr);
	vkFreeMemory(device, blockSumBuffer2Memory, nullptr);
	vkDestroyBuffer(device, tmpGaussianBuffer, nullptr);
	vkFreeMemory(device, tmpGaussianBufferMemory, nullptr);
	vkDestroyBuffer(device, tmpAdamBuffer, nullptr);
	vkFreeMemory(device, tmpAdamBufferMemory, nullptr);
	vkDestroyBuffer(device, dummyBuffer, nullptr);
	vkFreeMemory(device, dummyBufferMemory, nullptr);

	vkDestroyPipeline(device, densityControlPipeline, nullptr);
	vkDestroyPipelineLayout(device, densityControlPipelineLayout, nullptr);
	vkDestroyPipeline(device, markPrunePipeline, nullptr);
	vkDestroyPipelineLayout(device, markPrunePipelineLayout, nullptr);
	vkDestroyPipeline(device, scanLocalPipeline, nullptr);
	vkDestroyPipelineLayout(device, scanLocalPipelineLayout, nullptr);
	vkDestroyPipeline(device, scanAddPipeline, nullptr);
	vkDestroyPipelineLayout(device, scanAddPipelineLayout, nullptr);
	vkDestroyPipeline(device, compactPipeline, nullptr);
	vkDestroyPipelineLayout(device, compactPipelineLayout, nullptr);

	vkDestroyDescriptorPool(device, scanPool, nullptr);
	vkDestroyDescriptorSetLayout(device, scanSetLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, compactSpecificSetLayout, nullptr);
	vkDestroyImageView(device, outContributionImageView, nullptr);
	vkDestroyImageView(device, gtImageView, nullptr);
	vkDestroyImageView(device, curSSIMStatsImageView, nullptr);
	vkDestroyImageView(device, tempSSIMStatsImageView, nullptr);
	vkDestroyImageView(device, tempSSIMStatsImageView2, nullptr);

	vkDestroyImage(device, outContribution, nullptr);
	vkDestroyImage(device, gtImage, nullptr);
	vkDestroyImage(device, curSSIMStats, nullptr);
	vkDestroyImage(device, tempSSIMStats, nullptr);
	vkDestroyImage(device, tempSSIMStats2, nullptr);

	vkFreeMemory(device, outContributionMemory, nullptr);
	vkFreeMemory(device, gtImageMemory, nullptr);
	vkFreeMemory(device, curSSIMStatsMemory, nullptr);
	vkFreeMemory(device, tempSSIMStatsMemory, nullptr);
	vkFreeMemory(device, tempSSIMStatsMemory2, nullptr);

}

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
	maxGaussians = totalGaussians << 5;
	std::vector<Gaussian3D> src_tmp = gaussianFromPoints(points, totalGaussians, maxGaussians);

	createStorageBuffer<Gaussian3D>(src_tmp, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gaussianBuffer, gaussianBufferMemory);
	createStorageBuffer<Gaussian2D>(maxGaussians, projectedGaussianBuffer, projectedGaussianBufferMemory);
	createBuffer(sizeof(uint32_t),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gaussianCountBuffer,
				gaussianCountBufferMemory);
	{
		VkCommandBuffer cmdbuf = beginSingleTimeCommands();
		vkCmdFillBuffer(cmdbuf, gaussianCountBuffer, 0, sizeof(uint32_t), totalGaussians);
		endSingleTimeCommands(cmdbuf);
	}

	int n_tiles_row = (render_height + 15) >> 4;
	int n_tiles_col = (render_width + 15) >> 4;
	num_tiles = n_tiles_row * n_tiles_col;
	createStorageBuffer<TileRange>(num_tiles, tileRangeBuffer, tileRangeBufferMemory);
	createBuffer(sizeof(uint32_t) * 3,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
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
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreenImages[i], imageMemory[i]);
		createImageView(VK_FORMAT_R8G8B8A8_UNORM, offscreenImages[i], offscreenImageViews[i]);
	}
	createSorterAndBuffer();
	createDescriptorSetLayouts();
	createDescriptorPool();
	createDescriptorSets();

	VkPushConstantRange push{};
	push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	push.offset = 0;
	push.size = sizeof(ProjPush); // kvCapacity, n_cols, n_rows
	std::array<VkDescriptorSetLayout, 2> setLayout = {globalDescriptorSetLayout, localDescriptorSetLayout};
	createComputePipeline(projComputePipeline, projComputePipelineLayout,
						"../shader/spv/projection.spv", 1, &push,
						setLayout.size(), setLayout.data());
	VkPushConstantRange rangePush{};
	rangePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	rangePush.offset = 0;
	rangePush.size = sizeof(uint32_t); // num_tiles
	createComputePipeline(rangeComputePipeline, rangeComputePipelineLayout,
						"../shader/spv/find_range.spv", 1, &rangePush, 1,
						&globalDescriptorSetLayout);
	VkPushConstantRange rasterPush{};
	rasterPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	rasterPush.offset = 0;
	rasterPush.size = sizeof(ProjPush); // kvCapacity, n_cols, n_rows
	createComputePipeline(rasterComputePipeline, rasterComputePipelineLayout,
						"../shader/spv/raster.spv", 1, &rasterPush,
						setLayout.size(), setLayout.data());

	VkPushConstantRange argpassPush{};
	argpassPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	argpassPush.offset = 0;
	argpassPush.size = sizeof(uint32_t); // kvCapacity
	createComputePipeline(argpassComputePipeline, argpassComputePipelineLayout,
						"../shader/spv/argpass.spv", 1, &argpassPush,
						setLayout.size(), setLayout.data());

	createComputePipeline(argpass2Pipeline, argpass2PipelineLayout,
						"../shader/spv/argpass2.spv", 0, nullptr, 1,
						&globalDescriptorSetLayout);
	// createRenderpass();
}

Engine::Engine(uint64_t src_width, uint64_t src_height, float scale, std::vector<Core::Gaussian3D> &gaussians) {

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
	totalGaussians = gaussians.size();
	maxGaussians = totalGaussians;

	createStorageBuffer<Gaussian3D>(gaussians, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
									gaussianBuffer, gaussianBufferMemory);
	createStorageBuffer<Gaussian2D>(maxGaussians, projectedGaussianBuffer,
									projectedGaussianBufferMemory);
	createBuffer(sizeof(uint32_t),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gaussianCountBuffer,
				gaussianCountBufferMemory);
	{
		VkCommandBuffer cmdbuf = beginSingleTimeCommands();
		vkCmdFillBuffer(cmdbuf, gaussianCountBuffer, 0, sizeof(uint32_t), totalGaussians);
		endSingleTimeCommands(cmdbuf);
	}

	int n_tiles_row = (render_height + 15) >> 4;
	int n_tiles_col = (render_width + 15) >> 4;
	num_tiles = n_tiles_row * n_tiles_col;
	createStorageBuffer<TileRange>(num_tiles, tileRangeBuffer, tileRangeBufferMemory);
	createBuffer(sizeof(uint32_t) * 3,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
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
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
	push.size = sizeof(ProjPush); // kvCapacity, n_cols, n_rows
	std::array<VkDescriptorSetLayout, 2> setLayout = {globalDescriptorSetLayout, localDescriptorSetLayout};
	createComputePipeline(projComputePipeline, projComputePipelineLayout,
						"../shader/spv/projection.spv", 1, &push,
						setLayout.size(), setLayout.data());
	VkPushConstantRange rangePush{};
	rangePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	rangePush.offset = 0;
	rangePush.size = sizeof(uint32_t); // num_tiles
	createComputePipeline(rangeComputePipeline, rangeComputePipelineLayout,
						"../shader/spv/find_range.spv", 1, &rangePush, 1,
						&globalDescriptorSetLayout);
	VkPushConstantRange rasterPush{};
	rasterPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	rasterPush.offset = 0;
	rasterPush.size = sizeof(ProjPush); // kvCapacity, n_cols, n_rows
	createComputePipeline(rasterComputePipeline, rasterComputePipelineLayout,
						"../shader/spv/raster.spv", 1, &rasterPush,
						setLayout.size(), setLayout.data());

	VkPushConstantRange argpassPush{};
	argpassPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	argpassPush.offset = 0;
	argpassPush.size = sizeof(uint32_t); // kvCapacity
	createComputePipeline(argpassComputePipeline, argpassComputePipelineLayout,
						"../shader/spv/argpass.spv", 1, &argpassPush,
						setLayout.size(), setLayout.data());

	createComputePipeline(argpass2Pipeline, argpass2PipelineLayout,
						"../shader/spv/argpass2.spv", 0, nullptr, 1,
						&globalDescriptorSetLayout);
}

Engine::~Engine() {
	vkDeviceWaitIdle(device);
	// vkDestroyRenderPass(device, renderpass, nullptr);
	vkDestroyPipeline(device, argpass2Pipeline, nullptr);
	vkDestroyPipeline(device, argpassComputePipeline, nullptr);
	vkDestroyPipeline(device, rasterComputePipeline, nullptr);
	vkDestroyPipeline(device, rangeComputePipeline, nullptr);
	vkDestroyPipeline(device, projComputePipeline, nullptr);
	vkDestroyPipelineLayout(device, argpass2PipelineLayout, nullptr);
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
	vkDestroyBuffer(device, gaussianCountBuffer, nullptr);

	vkFreeMemory(device, keyBufferMemory, nullptr);
	vkFreeMemory(device, valueBufferMemory, nullptr);
	vkFreeMemory(device, counterBufferMemory, nullptr);
	vkFreeMemory(device, pingpongBufferMemory, nullptr);
	vkFreeMemory(device, tileRangeBufferMemory, nullptr);
	vkFreeMemory(device, indirectArgsBufferMemory, nullptr);
	vkFreeMemory(device, gaussianBufferMemory, nullptr);
	vkFreeMemory(device, projectedGaussianBufferMemory, nullptr);
	vkFreeMemory(device, gaussianCountBufferMemory, nullptr);

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
  	if (!enableValidationLayers) return;

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
		if (fmt.format == VK_FORMAT_R8G8B8A8_SRGB &&
			fmt.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {

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

	KVCapacity = maxGaussians << 8; // static pre-allocation pool (reverted from 10 to 8 to fix Apple M1 Out of Memory crash)

	// VkDeviceSize keyBufferSize = totalKVCount * sizeof(uint64_t);
	// VkDeviceSize valueBufferSize = totalKVCount * sizeof(uint32_t);
	VkDeviceSize counterBufferSize = sizeof(uint32_t);

	VrdxSorterStorageRequirements reqs;
	vrdxGetSorterKeyValueStorageRequirements(sorter, KVCapacity, &reqs);

	VkDeviceSize kvBufferSize = std::max(reqs.size, (VkDeviceSize)(KVCapacity * sizeof(uint32_t)));
	VkBufferUsageFlags kvUsage = reqs.usage |
								VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
								VK_BUFFER_USAGE_TRANSFER_DST_BIT |
								VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	createBuffer(kvBufferSize, kvUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, keyBuffer, keyBufferMemory);
	createBuffer(kvBufferSize, kvUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, valueBuffer, valueBufferMemory);
	createBuffer(reqs.size, reqs.usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pingpongBuffer,
				pingpongBufferMemory);
	createBuffer(
		counterBufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, counterBuffer, counterBufferMemory);
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
	std::array<VkDescriptorSetLayoutBinding, 8> globalBindings{};
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

	globalBindings[7].binding = 7; // gaussian counter
	globalBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	globalBindings[7].descriptorCount = 1;
	globalBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	globalBindings[7].pImmutableSamplers = nullptr;

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
	poolSizes[0] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8};
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

	// 1. global descriptor set
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &globalDescriptorSetLayout;
	
	vkAllocateDescriptorSets(device, &allocInfo, &globalDescriptorSets);

	VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = gaussianBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = VK_WHOLE_SIZE;

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

	VkDescriptorBufferInfo bufferInfo7{};
	bufferInfo7.buffer = gaussianCountBuffer;
	bufferInfo7.offset = 0;
	bufferInfo7.range = VK_WHOLE_SIZE;

	std::array<VkWriteDescriptorSet, 8> descriptorWriteGlobal{};
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
	descriptorWriteGlobal[7] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = globalDescriptorSets,
		.dstBinding = 7,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pImageInfo = nullptr,
		.pBufferInfo = &bufferInfo7,
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

		vkUpdateDescriptorSets(device, descriptorWritesLocal.size(), descriptorWritesLocal.data(), 0, nullptr);
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
                          VkMemoryPropertyFlags properties, VkBuffer &buffer,
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

void Engine::createImage(uint32_t width, uint32_t height, VkFormat imageFormat,
                         VkImageUsageFlags imageUsage,
                         VkMemoryPropertyFlags properties, VkImage &image,
                         VkDeviceMemory &imageMemory) {

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
	createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);
}

template <typename T>
void Engine::createStorageBuffer(std::vector<T> &srcBuffer, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {

	VkDeviceSize size = sizeof(T) * srcBuffer.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	void *pData;

	createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

	vkMapMemory(device, stagingBufferMemory, 0, size, 0, &pData);
	memcpy(pData, srcBuffer.data(), size);
	vkUnmapMemory(device, stagingBufferMemory);

	createBuffer(size,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

	VkBufferCopy region{};
	region.size = size;

	VkCommandBuffer cmdbuf = beginSingleTimeCommands();
	vkCmdCopyBuffer(cmdbuf, stagingBuffer, buffer, 1, &region);
	endSingleTimeCommands(cmdbuf);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}

template <typename T>
void Engine::createStorageBuffer(std::vector<T> &srcBuffer,
                                 VkBufferUsageFlags usage, VkBuffer &buffer,
                                 VkDeviceMemory &bufferMemory) {

	VkDeviceSize size = sizeof(T) * srcBuffer.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	void *pData;

	createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer, stagingBufferMemory);

	vkMapMemory(device, stagingBufferMemory, 0, size, 0, &pData);
	memcpy(pData, srcBuffer.data(), size);
	vkUnmapMemory(device, stagingBufferMemory);

	createBuffer(size,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

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

void Engine::endSingleTimeComputeCommands(VkCommandBuffer commandBuffer) {
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(computeQueue);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
};

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

	// Use fixed world up for manual control to prevent rolling and slanted movement
	const glm::vec3 worldUp = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 right = glm::normalize(glm::cross(cameraFront, worldUp));

	if (glfwGetKey(pWindow, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += moveSpeed * cameraFront;
	if (glfwGetKey(pWindow, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= moveSpeed * cameraFront;
	if (glfwGetKey(pWindow, GLFW_KEY_SPACE) == GLFW_PRESS)
		cameraPos += moveSpeed * worldUp;
	if (glfwGetKey(pWindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cameraPos -= moveSpeed * worldUp;
	if (glfwGetKey(pWindow, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= right * moveSpeed;
	if (glfwGetKey(pWindow, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += right * moveSpeed;

	if (glfwGetKey(pWindow, GLFW_KEY_UP) == GLFW_PRESS)
		pitch -= rotSpeed;
	if (glfwGetKey(pWindow, GLFW_KEY_DOWN) == GLFW_PRESS)
		pitch += rotSpeed;
	if (glfwGetKey(pWindow, GLFW_KEY_LEFT) == GLFW_PRESS)
		yaw += rotSpeed;
	if (glfwGetKey(pWindow, GLFW_KEY_RIGHT) == GLFW_PRESS)
		yaw -= rotSpeed;

	// Normalize angles
	if (pitch > 89.0f) pitch = 89.0f;
	if (pitch < -89.0f) pitch = -89.0f;
	yaw = fmod(yaw, 360.0f);

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);

	updateCamera(camUBO, cameraPos, cameraPos + cameraFront, render_width, render_height, worldUp);
	memcpy(cameraBufferMapped[currentFrame], &camUBO, sizeof(CameraUBO));
}

void Engine::setCameraFromColmap(const Image &image, const Camera *camera) {
	glm::quat q = image.q;
	glm::mat3 R_cw = glm::mat3_cast(q);
	glm::vec3 t_cw = image.t;

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

	// Intrinsics (Focal length)
	float fx = -1.0f, fy = -1.0f;
	if (camera) {
		float scaleX = (float)render_width / (float)camera->width;
		float scaleY = (float)render_height / (float)camera->height;
		fx = (float)camera->params[0] * scaleX;
		fy = (camera->model == 0) ? fx : (float)camera->params[1] * scaleY;
	}

	// Force update immediately
	updateCamera(camUBO, cameraPos, cameraPos + cameraFront, render_width, render_height, cameraUp, fx, fy);
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

void Engine::transitionImageLayout(VkImage image, VkFormat format,
                                   VkImageLayout oldLayout,
                                   VkImageLayout newLayout) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	endSingleTimeCommands(commandBuffer);
}

void Engine::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = {0, 0, 0};
	region.imageExtent = {width, height, 1};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(commandBuffer);
}

void Engine::verifyRadixSort(bool preSort) {
	// 1. Read element count from counterBuffer
	uint32_t count = 0;
	{
		VkBuffer stagingCount;
		VkDeviceMemory stagingCountMemory;
		createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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
	if (count == 0) return;
	if (count > KVCapacity) count = KVCapacity;

	auto checkBuffer = [&](VkBuffer buf, const char *name) {
		VkBuffer stagingKeys;
		VkDeviceMemory stagingKeysMemory;
		VkDeviceSize bufferSize = count * sizeof(uint32_t);
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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
			fprintf(stderr, "[Verify SUCCESS] Buffer '%s' is SORTED (%u keys)\n", name, count);
			fprintf(stderr, "    Sample[0..4] Keys:     %u, %u, %u, %u, %u\n", keys[0], keys[1], keys[2], keys[3], keys[4]);

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
			fprintf(stderr, "    Sample around failure: [%u]=%u, [%u]=%u, [%u]=%u\n",
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

float Engine::calculateSceneExtent(std::vector<Image> &images) {
	if (images.empty()) return 1.0f;

	glm::vec3 c_total(0.0f);
	std::vector<glm::vec3> camera_centers;
	camera_centers.reserve(images.size());

	for (const Image &image : images) {
		glm::mat3 Rt = glm::transpose(glm::mat3_cast(image.q));

		// 실제 카메라 위치 C = -Rt * t
		glm::vec3 t = {image.t[0], image.t[1], image.t[2]};
		glm::vec3 C = -Rt * t;

		camera_centers.push_back(C);
		c_total += C;
	}
	glm::vec3 scene_center = c_total / static_cast<float>(images.size());

	float max_dist = 0.0f;
	for (const glm::vec3 &C : camera_centers) {
		max_dist = std::max(max_dist, glm::distance(C, scene_center));
	}

	return max_dist * 1.1f;
}

} // namespace Core
