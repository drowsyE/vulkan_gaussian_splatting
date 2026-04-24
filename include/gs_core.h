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
    glm::quat q; // <- quaternion (rotation) , [w, x, y, z]
    glm::vec3 t; // <- translation
    uint32_t camera_id;
    std::string name;
} Image;

typedef struct Point {
    uint64_t id;
    double x, y, z;
    uint8_t r, g, b;
    double error;
} Point;

typedef struct Gaussian3D {
    glm::vec3 pos;       
    float pad1;          
    glm::vec4 scaleOpacity; // x, y, z, opacity
    glm::quat rot;          // quaternion [x, y, z, w]
    float sh[48];           // Spherical Harmonics coefficients (16 * 3 = 48)
    float pad2[4];          // Padding to 256 bytes tightly
} Gaussian3D;

typedef struct Gaussian2D { // Gaussians projected into screen plane
    glm::vec2 pos2d;            
    float pad0[2];                
    glm::vec3 conics;             
    float opacity;                
    glm::vec3 color;             
    float pad1;                   
} Gaussian2D ; 

typedef struct TileRange {
    uint32_t start;
    uint32_t end;
} TileRange;

typedef struct ProjPush {
	uint32_t capacity;
	uint32_t n_cols;
	uint32_t n_rows;
    uint32_t sh_degree;
} ProjPush;

typedef struct RasterPush {
	uint32_t capacity;
	float bgR;
	float bgG;
	float bgB;
	uint32_t n_cols;
	uint32_t n_rows;
} RasterPush;

typedef struct BackwardPush {
	float lambda;
	float bgR;
	float bgG;
	float bgB;
	uint32_t kvCapacity;
	uint32_t n_cols;
	uint32_t n_rows;
    uint32_t sh_degree;
} BackwardPush;

typedef struct DensityControlPush {
    float sceneExtent;
    uint32_t maxGaussians;
    uint32_t step;
} DensityControlPush;


// structure of cameras.bin : [number of camera] -> [camera1] -> [camera2] -> ...
std::vector<Camera> readCameras(const char *path);
std::vector<Image> readImages(const char *path);
std::vector<Point> readPoints(const char *path);
std::vector<Gaussian3D> gaussianFromPoints(std::vector<Point>& points, size_t size, size_t capacity);
void exportGaussians(const char *path, const std::vector<Gaussian3D>& gaussians, uint32_t num_gaussians);
std::vector<Gaussian3D> readGaussians(const char *path);

} // namespace Core