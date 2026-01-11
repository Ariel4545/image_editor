![Image editor (new logo)](https://github.com/Ariel4545/image_editor/assets/95249974/dbe7ac56-e72a-424e-8b9a-ab55454fcd0c)

# Image Editor

A simple yet powerful image editor built with Python, using **PySimpleGUI** for the interface and **Pillow** for image processing.

## Features ✨

The editor is organized into five main tabs for easy access to all tools:

### 🎨 Effects
*   **Blur**: Apply Gaussian blur to soften the image.
*   **Pixelate**: Pixelate the image for a retro or censored look.
*   **Posterize**: Reduce the number of colors for a poster-like effect.
*   **Solarize**: Invert pixel values above a threshold.
*   **Threshold**: Convert image to pure black and white based on intensity.

### 🖼️ Filters
*   **Detail**: Enhance fine details.
*   **Edge Enhance**: Emphasize edges within the image.
*   **Emboss**: Apply an emboss effect for a 3D look.
*   **Contour**: Extract and display the contours of the image.
*   **Invert**: Invert colors (negative effect).
*   **Sepia**: Apply a vintage sepia tone.
*   **Grayscale**: Convert the image to shades of gray.

### 🛠️ Adjustments
*   **Contrast**: Adjust the difference between light and dark areas.
*   **Brightness**: Make the image lighter or darker.
*   **Gamma**: Adjust gamma correction for non-linear brightness control.
*   **Sharpness**: Sharpen the image to make it clearer.

### 🌈 Color
*   **Temperature**: Adjust the warmth (red/blue balance) of the image.
*   **Saturation**: Adjust color intensity (from black & white to vibrant).
*   **RGB Balance**: Fine-tune Red, Green, and Blue channels individually.

### 🔄 Transform
*   **Flip X / Y**: Mirror the image horizontally or vertically.
*   **Rotation**: Rotate the image 360 degrees.
*   **Scale**: Resize the image (10% - 200%).
*   **Crop**: Crop the image from Left, Right, Top, and Bottom.

### ⚙️ Controls
*   **Save**: Save your edited image as PNG or JPEG.
*   **Reset**: Instantly revert all changes to default.

## Installation 📦

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install Pillow PySimpleGUI
```

## Usage 🚀

1.  Run the `main.py` script.
2.  Select an image file when prompted.
3.  Use the tabs to apply various effects and adjustments.
4.  Click **Save** to save your edited image as a PNG or JPEG file.
5.  Click **Reset** to clear all current edits.

## Roadmap 🗺️

*   [x] Crop functionality
*   [x] Reset button
*   [ ] Undo/Redo support
*   [ ] Batch processing
