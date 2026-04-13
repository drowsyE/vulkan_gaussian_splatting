#include <glm/glm.hpp>

namespace Core {

typedef struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 viewProj;
    alignas(16) glm::vec4 camPos;     // 3DGS의 SH 계산이나 View Direction 계산에 필요
    alignas(8)  glm::vec2 viewportSize;
    alignas(4)  float focalX;
    alignas(4)  float focalY;
} CameraUBO;

void updateCamera(CameraUBO& ubo, glm::vec3 position, glm::vec3 lookAt, float width, float height, glm::vec3 up);

}