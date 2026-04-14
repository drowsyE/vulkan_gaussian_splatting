#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace Core {

void updateCamera(CameraUBO& ubo, glm::vec3 position, glm::vec3 lookAt, float width, float height, glm::vec3 up) {
    // 1. View Matrix (카메라 위치와 방향)
    ubo.view = glm::lookAt(position, lookAt, up);

    // 2. Projection Matrix (원근감)
    // 3DGS 논문 기준으로는 fovX, fovY를 통해 계산된 투영 행렬을 사용합니다.
    float fovY = glm::radians(45.0f);
    ubo.proj = glm::perspective(fovY, width / height, 0.1f, 10000.0f);
    
    // Vulkan은 OpenGL과 달리 Y축이 반대이므로 보정 (필요 시)
    ubo.proj[1][1] *= -1;

    // 3. 편의를 위해 미리 곱해둔 행렬
    ubo.viewProj = ubo.proj * ubo.view;

    // 4. 기타 정보
    ubo.camPos = glm::vec4(position, 1.0f);
    ubo.viewportSize = glm::vec2(width, height);
    
    // 초점 거리(Focal Length) 계산 (3DGS 투영 시 필수)
    ubo.focalY = height / (2.0f * tan(fovY * 0.5f));
    ubo.focalX = ubo.focalY; // 보통은 동일하게 설정
}

}