![Image editor (new logo)](https://github.com/Ariel4545/image_editor/assets/95249974/dbe7ac56-e72a-424e-8b9a-ab55454fcd0c)

# Image Editor

A simple yet powerful image editor built with Python, using **Tkinter** for the interface and **Pillow** for image processing.

## Features ✨

The editor is organized into seven main tabs for easy access to all tools:

### 🎨 Effects
*   **Blur**: Apply Gaussian blur to soften the image.
*   **Pixelate**: Pixelate the image for a retro or censored look.
*   **Posterize**: Reduce the number of colors for a poster-like effect.
*   **Solarize**: Invert pixel values above a threshold.
*   **Threshold**: Convert image to pure black and white based on intensity.
*   **Vignette**: Add a dark gradient to the corners to focus on the center.
*   **Sepia**: Apply a vintage sepia tone with adjustable intensity.

### 🖼️ Filters
*   **Detail**: Enhance fine details.
*   **Edge Enhance**: Emphasize edges within the image.
*   **Emboss**: Apply an emboss effect for a 3D look.
*   **Contour**: Extract and display the contours of the image.
*   **Invert**: Invert colors (negative effect).
*   **Grayscale**: Convert the image to shades of gray.
*   **Auto Contrast**: Automatically adjust contrast.
*   **Equalize**: Apply histogram equalization.

### 🛠️ Adjustments
*   **Contrast**: Adjust the difference between light and dark areas.
*   **Brightness**: Make the image lighter or darker.
*   **Gamma**: Adjust gamma correction for non-linear brightness control.
*   **Sharpness**: Sharpen the image to make it clearer.

### 🌈 Color
*   **Temperature**: Adjust the warmth (red/blue balance) of the image.
*   **Hue**: Shift the hue of the image to change colors completely.
*   **Saturation**: Adjust color intensity (from black & white to vibrant).
*   **RGB Balance**: Fine-tune Red, Green, and Blue channels individually.

### 🔄 Transform
*   **Flip X / Y**: Mirror the image horizontally or vertically.
*   **Rotation**: Rotate the image 360 degrees.
    *   **Quick Rotate**: Buttons for -90° and +90° rotation.
*   **Scale**: Resize the image (10% - 200%).
*   **Crop**: Crop the image from Left, Right, Top, and Bottom.

### 🖼️ Border
*   **Size**: Add a solid border around the image.
*   **Color**: Customize the border color.

### ✍️ Text
*   **Content**: Add custom text overlay.
*   **Font Settings**: Adjust Size, Opacity, and Color.
*   **Position**: Place text precisely using X and Y coordinates.

### ⚙️ Controls
*   **Undo / Redo**: Revert or re-apply changes (History support).
*   **Presets**: Save and Load your favorite filter settings.
*   **Batch Process**: Apply current settings to an entire folder of images.
*   **Save**: Save your edited image as PNG or JPEG.
*   **Reset**: Instantly revert all changes to default.
*   **About**: View application version and dependency information.

## Installation 📦

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install Pillow
```

*Note: Tkinter is usually included with Python installations.*

## Usage 🚀

1.  Run the `main.py` script.
2.  Select an image file when prompted.
3.  Use the tabs to apply various effects and adjustments.
4.  Use **Undo/Redo** to experiment.
5.  Save your settings with **Save Preset** if you like them.
6.  Use **Batch Process** to apply the current look to multiple images.
7.  Click **Save** to save your edited image as a PNG or JPEG file.

## Roadmap 🗺️

*   [x] Crop functionality
*   [x] Reset button
*   [x] Undo/Redo support
*   [x] Text Overlay
*   [x] Save/Load Presets
*   [x] Batch processing
*   [x] Auto-enhancement tools
*   [x] Border tool
*   [x] High DPI Support
*   [x] Migrate to Tkinter (Remove PySimpleGUI dependency)
