#pragma once
#include <glm/glm.hpp>

struct AdamState {
    glm::vec3 pos_m;
    float pad1;
    glm::vec3 pos_v;
    float pad2;
    glm::vec4 scaleOpacity_m;
    glm::vec4 scaleOpacity_v;
    glm::vec4 rot_m;
    glm::vec4 rot_v;
    float sh_m[48];
    float sh_v[48];
    float pad3[8]; // Padding to exactly 512 bytes
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
    uint32_t sh_degree; // 현재 사용 중인 SH Degree
};


