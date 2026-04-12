#include "gs_core.h"

namespace Core {

// structure of cameras.bin : [number of camera] -> [camera1] -> [camera2] -> ...
std::vector<Camera> readCameras(const char *path) {
    std::ifstream file(path, std::ios::binary);

    uint64_t num_cams;
    file.read((char *)&num_cams, sizeof(num_cams));

    std::vector<Camera> cameras;
    for (int i = 0; i < num_cams; i++) {
        Camera camera;
        file.read((char *)&camera.id, sizeof(camera.id));
        file.read((char *)&camera.model, sizeof(camera.model));
        file.read((char *)&camera.width, sizeof(camera.width));
        file.read((char *)&camera.height, sizeof(camera.height));

        int num_params = (camera.model == 0) ? 3 : 4;
        // 0 -> simple pinhole(f, cx, cy), 1 -> pinhole(fx, fy, cx, cy)
        
        camera.params.resize(num_params);

        file.read((char *)camera.params.data(), sizeof(double) * num_params);

        cameras.push_back(camera);
    }

    return cameras;
}

std::vector<Image> readImages(const char *path) {
    std::ifstream file(path, std::ios::binary);

    uint64_t num_images;
    file.read((char *)&num_images, sizeof(num_images));

    std::vector<Image> images;
    for (int i = 0; i < num_images; i++) {
        Image img;
        file.read((char *)&img.image_id, sizeof(img.image_id));
        file.read((char *)img.q, sizeof(img.q));
        file.read((char *)img.t, sizeof(img.t));
        file.read((char *)&img.camera_id, sizeof(img.camera_id));

        char c;
        while (file.get(c) && c != '\0') {
        img.name += c;
        }

        // skip points2D
        uint64_t num_points2D;
        file.read((char *)&num_points2D, sizeof(num_points2D));
        file.ignore(num_points2D *
                    (2 * sizeof(double) + sizeof(int64_t))); // x, y, id

        images.push_back(img);
    }

    return images;
}

std::vector<Point> readPoints(const char *path) {
    std::ifstream file(path, std::ios::binary);

    uint64_t num_points;
    file.read((char *)&num_points, sizeof(num_points));

    std::vector<Point> points;
    for (int i = 0; i < num_points; i++) {
        Point p;
        file.read((char *)&p.id, sizeof(uint64_t));
        file.read((char *)&p.x, sizeof(double));
        file.read((char *)&p.y, sizeof(double));
        file.read((char *)&p.z, sizeof(double));
        file.read((char *)&p.r, sizeof(uint8_t));
        file.read((char *)&p.g, sizeof(uint8_t));
        file.read((char *)&p.b, sizeof(uint8_t));
        file.read((char *)&p.error, sizeof(double));

        uint64_t track_len; // number of pairs
        file.read((char *)&track_len, sizeof(track_len));

        // track skip
        file.ignore(track_len * sizeof(uint32_t) * 2); // (image_id, point2d_idx)

        points.push_back(p);
    }

    return points;
}

// std::vector<Gaussian3D> gaussianFromPoints(const char *path) {
//     std::ifstream file(path, std::ios::binary);

//     uint64_t num_points;
//     file.read((char *)&num_points, sizeof(num_points));

//     std::vector<Gaussian3D> gaussians;
//     Point p;
//     for (int i = 0; i < num_points; i++) {
//         file.read((char *)&p.id, sizeof(uint64_t));
//         file.read((char *)&p.x, sizeof(double));
//         file.read((char *)&p.y, sizeof(double));
//         file.read((char *)&p.z, sizeof(double));
//         file.read((char *)&p.r, sizeof(uint8_t));
//         file.read((char *)&p.g, sizeof(uint8_t));
//         file.read((char *)&p.b, sizeof(uint8_t));
//         file.read((char *)&p.error, sizeof(double));

//         uint64_t track_len; // number of pairs
//         file.read((char *)&track_len, sizeof(track_len));

//         // track skip
//         file.ignore(track_len * sizeof(uint32_t) * 2); // (image_id, point2d_idx)

//         Gaussian3D g;
//         g.pos = glm::vec3(p.x, p.y, p.z);
//         g.color = glm::vec3(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f);
//         g.scaleOpacity = glm::vec4(0.01f, 0.01f, 0.01f, 1.0f);
//         g.rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

//         gaussians.push_back(g);
//     }

//     return gaussians;
// }

std::vector<Gaussian3D> gaussianFromPoints(std::vector<Point> &points, size_t size, size_t capacity) {

    std::vector<Gaussian3D> gaussians(capacity);
    for (int i = 0; i < size; i++) {
        Point p = points[i];
        Gaussian3D g;
        g.pos = glm::vec3(p.x, p.y, p.z);
        g.color = glm::vec3(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f);
        g.scaleOpacity = glm::vec4(0.01f, 0.01f, 0.01f, 1.0f);
        g.rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

        gaussians.push_back(g);
    }

    return gaussians;
}

} // namespace Core