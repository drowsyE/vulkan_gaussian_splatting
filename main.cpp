#include "engine.h"
#include "gs_core.h"
#include <cstdlib>

int main() {

    // // Sparse reconstruction via COLMAP
    // // 1. feature extraction
    // system("colmap feature_extractor --database_path ../database.db --image_path ../images");
    // // 2. feature matching
    // system("colmap exhaustive_matcher --database_path ../database.db");
    // // 3. mapping
    // system("colmap mapper --database_path ../database.db --image_path ../images --output_path ../sparse");
    // // 4. undistort images
    // system("colmap image_undistorter --image_path ../images --input_path ../sparse/0 --output_path ../dense");

    std::vector<Core::Camera> cameras = Core::readCameras("../sparse/0/cameras.bin");
    std::vector<Core::Image> images = Core::readImages("../sparse/0/images.bin");
    std::vector<Core::Point> points = Core::readPoints("../sparse/0/points3D.bin");
    
    Core::Engine(cameras[0].width, cameras[0].height, 0.5, points);


}