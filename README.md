# Canny Edge Detection with Virtual Hexagonal Image Structure

## Overview
This project implements an enhanced version of the Canny edge detection algorithm by applying it to a virtual hexagonal image structure. Traditional image processing relies on square grids, which suffer from ambiguous connectivity and distance anisotropy. By simulating a hexagonal grid, this application benefits from uniform connectivity (6 equidistant neighbors) and better circular symmetry, resulting in superior edge detection. 

A key innovation in this project is the use of an optimized **linear interpolation** method for square-to-hexagonal grid conversion, which significantly reduces computational complexity compared to traditional bi-linear or tri-linear methods.

## Key Features
* **Virtual Hexagonal Grid Simulation:** Converts standard rectangular images into a virtual hexagonal structure by dividing square pixels into a 7x7 sub-pixel matrix, grouping 56 sub-pixels to form a single hexagonal pixel.
* **Optimized Linear Interpolation:** Uses a simplified 1D linear interpolation approach for grid conversion, cutting execution time by approximately 50% compared to previous complex methods.
* **Adapted Canny Pipeline:** Custom implementations of Gaussian filtering, Sobel operator, Non-Maximum Suppression (NMS), and Hysteresis thresholding, all mathematically adapted for the 3 axes of symmetry in a 6-neighbor geometry.
* **Interactive GUI:** Features a desktop interface to load images, dynamically adjust low and high hysteresis thresholds, and visually compare the custom hexagonal Canny results against the standard square-grid Canny algorithm.

## Technologies Used
The system is built with a modular architecture in Python:
* **Python:** Core programming language.
* **NumPy:** Handles images as multi-dimensional arrays and enables highly efficient, vectorized mathematical operations to avoid slow pixel-by-pixel iteration.
* **OpenCV:** Used for basic I/O operations (reading/writing images), color space conversions (RGB to Grayscale), and providing the baseline standard Canny implementation.
* **PySide6 (Qt):** Framework used to build the interactive Graphical User Interface (GUI).

## Results and Advantages
Extensive visual testing demonstrates that the hexagonal topology provides significant advantages:
1. **Superior Geometry:** Curves and diagonals are represented much more accurately, visibly reducing the "staircase" effect inherent to square grids.
2. **Noise Suppression:** The adapted Gaussian filter acts uniformly in all directions on the hexagonal grid, producing cleaner and more continuous contour maps.

## Installation and Usage

### Prerequisites
Ensure you have **Python 3.11** installed on your system to ensure full compatibility with the virtual environment.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/cristibutica/Canny_BEG.git
   cd Canny_BEG
   ```

2. Create and activate a virtual environment:
   * **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     python3.11 -m venv venv
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Ensure your virtual environment is activated, then launch the GUI by running the main entry point:
```bash
python main.py
```
From the interface, you can load images and use the sliders to adjust the thresholds and view the real-time edge map generation.

## Contributors
* Cristian Butica
* Tudor Erdei
* Mădălin Gavrilaș

*Project developed for the "Prelucrarea Numerică a Imaginiilor" (Digital Image Processing) course at the Technical University of Cluj-Napoca (UTCN), Faculty of Electronics, Telecommunications and Information Technology.*