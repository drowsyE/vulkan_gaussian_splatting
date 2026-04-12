#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace Core {

typedef struct Camera {
    uint32_t id;
    uint32_t model;
    uint64_t width;
    uint64_t height;
    std::vector<double> params; // camera instrinsic params
} Camera;

typedef struct Image {
    uint32_t image_id;
    double q[4]; // <- quaternion (rotation)
    double t[3]; // <- translation
    uint32_t camera_id;
    std::string name;
} Image;

typedef struct Point {
    uint64_t id;
    double x, y, z;
    uint8_t r, g, b;
    double error;
} Point;

typedef struct Gaussian {
    glm::vec3 pos;
    alignas(16) glm::vec4 scaleOpacity; // scale.x, scale.y, scale.z, opacity
    glm::quat rot;
    glm::vec3 color;
} Gaussian;

typedef struct ProjectedGaussian {
    glm::vec2 pos2d;
    alignas(8) glm::vec3 conics; // inverse covariance의 상삼각 성분 (3개)
    float opacity;
    glm::vec3 color;
    bool flag;
    uint64_t key;
} ProjectedGaussian ;

// structure of cameras.bin : [number of camera] -> [camera1] -> [camera2] -> ...
std::vector<Camera> readCameras(const char *path);
std::vector<Image> readImages(const char *path);
std::vector<Point> readPoints(const char *path);
std::vector<Gaussian> gaussianFromPoints(std::vector<Point>& points, size_t size, size_t capacity);

} // namespace Core