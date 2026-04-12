#include "engine_utils.h"
#include <glm/gtc/matrix_transform.hpp>

namespace Core {

const char* getDebugSeverityStr(VkDebugUtilsMessageSeverityFlagBitsEXT Severity) {
    switch (Severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            return "Verbose";

        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            return "Info";

        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            return "Warning";

        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            return "Error";

        default:
            printf("Invalid severity code %d\n", Severity);
            throw std::runtime_error("");

    return "No such severity!";
  }
}

const char* getDebugType(VkDebugUtilsMessageTypeFlagsEXT Type) {
    switch (Type) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
            return "General";

        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
            return "Validation";

        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
            return "Performance";

        case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
            return "Device address binding";

        default:
            printf("Invalid type code %d\n", Type);
            throw std::runtime_error("");
  }
  return "No such type!";
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT Severity,
    VkDebugUtilsMessageTypeFlagsEXT Type,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *userData) {

    printf("[Info]\nDebug callback: %s\n", pCallbackData->pMessage);
    printf("Severity : %s\n", getDebugSeverityStr(Severity));
    printf("Type : %s\n", getDebugType(Type));
    printf("Objects : ");

    for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
        printf("%llx ", pCallbackData->pObjects[i].objectHandle);
    }
    printf("\n\n");

  return VK_FALSE;
}

void populateDebugMessenger(
    VkDebugUtilsMessengerCreateInfoEXT &debugMessengerCreateInfo) {
    debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

    debugMessengerCreateInfo.messageSeverity =
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    debugMessengerCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

    debugMessengerCreateInfo.pfnUserCallback = debugCallback;
}

VkResult createDebugUtilsMessenger(VkInstance &instance,
                          const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDebugUtilsMessengerEXT *pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator) {
    PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

std::vector<char> readFile(const char *filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkShaderModule createShader(VkDevice &device, const char* filename) {
    auto shaderCode = readFile(filename);
    VkShaderModule shader;
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t*> (shaderCode.data());
    vkCreateShaderModule(device, &shaderInfo, nullptr, &shader);
    return shader;
}

}