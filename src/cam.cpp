#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace Core {

void updateCamera(CameraUBO& ubo, glm::vec3 position, glm::vec3 lookAt, float width, float height, glm::vec3 up, float fx, float fy) {
    ubo.view = glm::lookAt(position, lookAt, up);

    float fovY;
    if (fx > 0 && fy > 0) {
        ubo.focalX = fx;
        ubo.focalY = fy;
        fovY = 2.0f * atan(height / (2.0f * fy));
    } else {
        fovY = glm::radians(45.0f);
        ubo.focalY = height / (2.0f * tan(fovY * 0.5f));
        ubo.focalX = ubo.focalY;
    }

    ubo.proj = glm::perspective(fovY, width / height, 0.1f, 10000.0f);
    
    ubo.proj[1][1] *= -1;

    ubo.viewProj = ubo.proj * ubo.view;

    ubo.camPos = glm::vec4(position, 1.0f);
    ubo.viewportSize = glm::vec2(width, height);
}

}