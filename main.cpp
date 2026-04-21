#include "engine.h"
#include "gs_core.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void printHelp(const char* execName) {
    std::cout << "Usage: " << execName << " [options]\n"
              << "Options:\n"
              << "  -h, --help           Show this help message\n"
              << "  -t, --train          Start training mode\n"
              << "  -i, --iter <n>       Number of training iterations (default: 100)\n"
              << "  -v, --view <path>    Load and view a pre-trained .bin file\n"
              << "  -c, --colmap         Run COLMAP reconstruction pipeline before starting\n"
              << "  -s, --scale <float>  Set rendering scale (default: 0.5)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    bool runColmap = false;
    bool trainMode = false;
    bool viewMode = false;
    std::string loadPath = "";
    int iterations = 100;
    float scale = 0.5f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        } else if (arg == "-t" || arg == "--train") {
            trainMode = true;
        } else if (arg == "-i" || arg == "--iter") {
            if (i + 1 < argc) iterations = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--view" || arg == "-l" || arg == "--load") {
            if (i + 1 < argc) {
                loadPath = argv[++i];
                viewMode = true;
            }
        } else if (arg == "-c" || arg == "--colmap") {
            runColmap = true;
        } else if (arg == "-s" || arg == "--scale") {
            if (i + 1 < argc) scale = std::stof(argv[++i]);
        }
    }

    if (runColmap) {
        // Sparse reconstruction via COLMAP
        std::cout << "Running COLMAP reconstruction pipeline..." << std::endl;

        // Initialize existing COLMAP files and directories
        fs::remove("../database.db");
        fs::remove_all("../sparse");
        fs::remove_all("../dense");

        system("colmap feature_extractor --database_path ../database.db --image_path ../images");
        system("colmap exhaustive_matcher --database_path ../database.db");
        std::string sparsePath = "../sparse";
        if (!fs::exists(sparsePath) || !fs::is_directory(sparsePath)) {
            fs::create_directories(sparsePath);
        }
        system("colmap mapper --database_path ../database.db --image_path ../images --output_path ../sparse");
        system("colmap image_undistorter --image_path ../images --input_path ../sparse/0 --output_path ../dense");
        return 0;
    }

    // Default smart behavior if no specific mode is chosen
    if (!trainMode && !viewMode) {
        loadPath = "trained_gaussians.bin";
        if (!fs::exists(loadPath)) {
            loadPath = "build/trained_gaussians.bin";
        }
        
        std::string datasetPath = "../dense/sparse/points3D.bin";
        
        if (fs::exists(loadPath)) {
            bool datasetUpdated = false;
            if (fs::exists(datasetPath)) {
                auto modelTime = fs::last_write_time(loadPath);
                auto datasetTime = fs::last_write_time(datasetPath);
                if (datasetTime > modelTime) {
                    datasetUpdated = true;
                }
            }
            
            if (datasetUpdated) {
                trainMode = true;
                std::cout << "Dataset has been updated recently. Starting training mode..." << std::endl;
            } else {
                viewMode = true;
                std::cout << "No arguments provided. Found default model: " << loadPath << std::endl;
            }
        } else {
            trainMode = true;
            std::cout << "No arguments provided and no default model found. Starting training mode..." << std::endl;
        }
    }

    std::vector<Core::Camera> cameras = Core::readCameras("../dense/sparse/cameras.bin");
    std::vector<Core::Image> images = Core::readImages("../dense/sparse/images.bin");

    if (viewMode) {
        if (loadPath.empty()) loadPath = "trained_gaussians.bin";

        if (fs::exists(loadPath)) {
            printf("Loading trained gaussians from %s...\n", loadPath.c_str());
            std::vector<Core::Gaussian3D> gaussians = Core::readGaussians(loadPath.c_str());
            Core::Engine engine(cameras[0].width, cameras[0].height, scale, gaussians);
            if (!images.empty()) {
                engine.setCameraFromColmap(images[0]);
            }
            engine.run();
            return 0;
        } else {
            std::cerr << "Error: File not found: " << loadPath << std::endl;
            return 1;
        }
    }

    // Training mode
    std::vector<Core::Point> points = Core::readPoints("../dense/sparse/points3D.bin");
    printf("Number of points : %zu\n", points.size());
    
    Core::Engine engine(cameras[0].width, cameras[0].height, scale, points);
    if (!images.empty()) {
        engine.setCameraFromColmap(images[0]);
    }
    
    std::cout << "Starting training for " << iterations << " iterations with scale " << scale << "..." << std::endl;
    engine.train(images, cameras, iterations);

    return 0;
}