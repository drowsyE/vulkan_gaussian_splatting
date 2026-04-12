#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <fstream>

namespace Core {

const char* getDebugSeverityStr(VkDebugUtilsMessageSeverityFlagBitsEXT Severity);

const char* getDebugType(VkDebugUtilsMessageTypeFlagsEXT Type);

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT Severity,
    VkDebugUtilsMessageTypeFlagsEXT Type,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *userData);

void populateDebugMessenger(
    VkDebugUtilsMessengerCreateInfoEXT &debugMessengerCreateInfo);

VkResult createDebugUtilsMessenger(VkInstance &instance,
                          const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDebugUtilsMessengerEXT *pDebugMessenger);

void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator);

std::vector<char> readFile(const char *filename);

VkShaderModule createShader(VkDevice &device, const char* filename);
}