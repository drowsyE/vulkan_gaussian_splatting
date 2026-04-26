# Vulkan Gaussian Splatting

Vulkan-based 3D Gaussian Splatting rendering and training engine built from scratch. It features a complete pipeline (forward rendering and backward training loops) and is specifically optimized to perform well on Apple Silicon (M1/M2/M3) using MoltenVK.

## Features

- **Training & Rendering**: Full 3D Gaussian Splatting pipeline including tile-based rasterization, GPU-accelerated Radix sorting, and density control.
- **Apple Silicon Optimized**: Stable 60 FPS rendering on Apple M1 chips, with tailored GPU synchronization strategies to bypass MoltenVK and TBDR architecture limitations.
- **6-DoF Navigation**: Interactive first-person 3D camera controls (Yaw, Pitch, Roll) to explore reconstructed scenes in real-time.
- **COLMAP Integration**: Built-in support to run the entire COLMAP pipeline (feature extraction, mapping, undistortion) to bootstrap scenes from raw images.

## Dependencies

*   **Vulkan** (version 1.3+ recommended)
*   **GLFW3**
*   **GLM**
*   **CMake**
*   **COLMAP** (required only if generating new sparse point clouds from raw images)

## Build Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/vulkan_gaussian_splatting.git
cd vulkan_gaussian_splatting

# Create build directory
mkdir build
cd build

# Compile
make
```

## Usage

The primary executable is `gaussian_splatting`. It supports several operating modes:

```bash
Usage: ./gaussian_splatting [options]

Options:
  -h, --help           Show help message
  -t, --train          Start training mode
  -i, --iter <n>       Number of training iterations (default: 100)
  -v, --view <path>    Load and view a pre-trained .bin file
  -c, --colmap         Run COLMAP reconstruction pipeline before starting
  -s, --scale <float>  Set rendering scale (default: 0.5)
```

### Typical Workflow

**1. Data Preparation**
Place your source images inside the `images/` directory at the project root.

**2. COLMAP Reconstruction**
Run the engine with the `-c` flag. It will automatically run COLMAP, extract features, and output the sparse/dense structures required for training.
```bash
./gaussian_splatting -c
```

**3. Training**
Run the engine in training mode. It will use the point cloud extracted from `dense/sparse/points3D.bin`.
```bash
./gaussian_splatting -t -i 3500
```

**4. Viewing**
Once trained, launch the engine with no arguments (it intelligently detects the newly built model) or specify the model directly:
```bash
./gaussian_splatting -v trained_gaussians.bin
```

## Scene Navigation Controls

- **W, A, S, D**: Move Forward, Left, Backward, Right
- **Q, E**: Roll Left / Right
- **Arrow Keys**: Rotate Camera (Yaw, Pitch)

## Project Structure

- `src/` & `include/`: Core engine code (`engine.cpp`, `gs_core.cpp`, `cam.cpp`).
- `shader/`: Vulkan GLSL shaders (`projection.comp`, `raster.comp`, `backward.comp`, `density_control.comp`).
- `lib/`: Third-party dependencies (e.g., `vk_radix_sort`).
- `walkthrough.md`: Developer notes on engine stabilizations and Vulkan workarounds.

## Results

### Lego (from Nerf Synthetic Dataset)
<img width="880" height="679" alt="lego1-7 3k" src="https://github.com/user-attachments/assets/37069394-40a6-409c-bd20-e2fdcd98adca" />
<img width="918" height="627" alt="lego2-7 3k" src="https://github.com/user-attachments/assets/c5ba65cf-1bdc-431b-9af9-3a14e0c9e889" />
<img width="873" height="563" alt="lego3-7 3k" src="https://github.com/user-attachments/assets/9d46b5b8-322c-410e-a19a-815bde00c391" />

### Drum (from Nerf Synthetic Dataset)
<img width="888" height="484" alt="drum" src="https://github.com/user-attachments/assets/3bc93c0e-0596-4a91-80f9-1464b5bc04fb" />

### Hotdog (from Nerf Synthetic Dataset)
<img width="941" height="693" alt="hotdog" src="https://github.com/user-attachments/assets/8e1e2162-3c0f-4465-a8e8-d470e0ff3feb" />
