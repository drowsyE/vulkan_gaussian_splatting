#pragma once
#include <glm/glm.hpp>

struct AdamState {
    glm::vec4 posR_m; // pos + color.R
    glm::vec4 posR_v;
    glm::vec4 scaleOpacity_m; // scale + opacity
    glm::vec4 scaleOpacity_v;
    glm::vec4 rot_m;
    glm::vec4 rot_v;
    glm::vec2 colorGB_m;
    glm::vec2 colorGB_v;
};

struct AdamPush {
    float pos_lr;       
    float rot_lr;
    float scale_lr;
    float opacity_lr;
    float color_lr;

    float beta1;        // 보통 0.9
    float beta2;        // 보통 0.999
    float epsilon;      // 보통 1e-8
    uint32_t step;      // 현재 트레이닝 스텝 (1부터 시작)
};


