# Image Denoise Filter Tool with Streamlit

A simple image processing web app built using **Streamlit** application designed to demonstrate various image processing techniques, including Gaussian Blur, Median Filtering, PSNR (Peak Signal-to-Noise Ratio), and SSIM (Structural Similarity Index). This app allows users to upload an image and apply these filters, viewing the processed images alongside their original counterpart. It also provides metrics to quantitatively compare the differences between the original and filtered images.  As well as, view histograms of RGB color channels for each image. 

---

## Table of Contents

  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
  - [Example Screenshot and Video](#example-screenshot-and-video)
  - [Technical Notes](#technical-notes)
  - [Filtering Techniques and Metrics Definitions](#filtering-techniques-and-metrics-definitions)
  - [MIT License](#mit-license)

---

## Project Overview

This project is a **web-based image processing application** built using **Streamlit**, allowing users to:

- Upload images in `.jpg`, `.jpeg`, or `.png` format.
- Apply Gaussian Blur and Median Filtering techniques.
- View processed images side-by-side with the original.
- Calculate metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).
- Save the processed images to disk.

It's ideal for educational purposes, image processing experiments, or quick prototyping in a web environment.
   
---

## Features

| Feature                   | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| ✅ Image Upload            | Supports JPG, JPEG, and PNG formats.                           |
| ✅ Gaussian Blur           | Apply blur with adjustable kernel size and sigma.              |
| ✅ Median Filter           | Apply noise reduction using median filtering.                  |
| ✅ PSNR & SSIM Metrics     | Calculate image quality metrics for comparison.                |
| ✅ Histogram Visualization | View RGB histograms of original, blurred, and filtered images. |
| ✅ Save Processed Images   | Export processed results to disk.                              |

---

## Installation

To run this application locally, follow these steps:

1. **Install dependencies** using pip:
   ```bash
   pip install streamlit numpy opencv-python pillow scikit-image matplotlib
   ```

2. **Save the code in a file**, for example: `app.py`.

3. **Run the app with Streamlit**:
   ```bash
   streamlit run app.py
   ```
---

## Usage

1. **Upload an image (JPG/JPEG/PNG)**: Use the file uploader to load an image in JPG, JPEG, or PNG format.
2. **Adjust Parameters for Image Processing**:
   - Adjust Gaussian Blur kernel size and sigma
   - Adjust Median Filter kernel size
3. **View Processed Results:** After adjusting, you'll see three side-by-side images:
   - Original
   - Gaussian Blurred Image
   - Median Filtered Image
4. **Compute Metrics**: The app will automatically calculate and display the following metrics for each processed image:
   - **PSNR** (Peak Signal-to-Noise Ratio) in dB
   - **SSIM** (Structural Similarity Index)
5. **Save Processed Images**:You can click "Save Processed Images" to save both results as JPEG files.
6. **View Color Histograms**: There's an option to view histograms of RGB color channels for each image. Display histograms of the RGB channels for the original, Gaussian-blurred, and median-filtered images.

---

## Code Structure

The code is organized into a single Python script (`app.py`) with the following key components:

- `ImageProcessor`: A class that handles image loading, processing, and metric computation.
- `main()`: The Streamlit app entry point that defines UI layout, input handling, and output rendering.

---

## Example Screenshot and Video


<video width="630" height="300" src="https://github.com/user-attachments/assets/f81f76ad-ca7b-4352-8cd7-b1649d09b020"></video>

---

## Technical Notes

- The application uses **OpenCV** for image processing and **Streamlit** for the web interface.
- Images are loaded as NumPy arrays, processed using OpenCV filters (Gaussian Blur & Median Filter), and displayed in RGB format.
- PSNR and SSIM metrics are computed to evaluate the quality of the processed images.
- Histograms are visualized using **Matplotlib**.

## Filtering Techniques and Metrics Definitions  

- **Gaussian Blur**  
   A smoothing filter that reduces noise by averaging pixel values with a Gaussian distribution kernel. Ideal for reducing fine details but can blur edges.  

- **Median Filter**  
   A non-linear filter that replaces each pixel with the median of its neighbors, effective against salt-and-pepper noise while preserving edges better than Gaussian filters.  

- **Peak Signal-to-Noise Ratio (PSNR)**  
**What it measures:**  
  PSNR quantifies the **quality of a distorted image** compared to its original counterpart. It evaluates how much noise or distortion has been introduced during processing, such as compression, filtering, or resizing.

**How it works:**  
  PSNR is calculated using the **Mean Squared Error (MSE)** between the original and processed images.  
- **The formula**:  
  $\text{PSNR} = 10\cdot\log_{10}\left(\frac{\text{MAX\_I}^2}{\text{MSE}}\right)$
    
  where `MAX_I` is the maximum pixel value (e.g., 255 for 8-bit images).  

**What it tells you:**  
- **Higher PSNR = Better Quality**: Indicates minimal noise or distortion.  
- **Lower PSNR = More Distortion**: Suggests significant degradation in image quality (e.g., due to compression, blurring, or filtering).
  
- **Use Case**: Often used for objective evaluation of image compression algorithms or noise reduction techniques.  

---

- **Structural Similarity Index (SSIM)**  
**What it measures:**  
  SSIM evaluates how similar two images are in terms of their **structure, luminance (brightness), and contrast**, mimicking human visual perception. It is more aligned with subjective image quality assessment than PSNR.

**How it works:**  
- SSIM compares the **local structure** of patches in both images using a windowed approach.  
- **The formula**:  
  $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ 

  where `μ` is the mean, `σ` is the standard deviation, and `C₁`, `C₂` are constants to stabilize values.  

**What it tells you:**  
- **Higher SSIM = Better Structural Similarity**: Indicates that the processed image retains key structural details of the original (e.g., edges, textures).  
- **Lower SSIM = Structural Differences**: Suggests significant changes in visual structure (e.g., blurring, edge loss, or artifacts).  
- **Use Case**: Ideal for assessing perceptual quality in applications like image restoration, compression, and filtering.  

---

### **Key Differences:**
| **Metric**   | **Focus**              | **What It Tells You**                          |
|--------------|------------------------|------------------------------------------------|
| **PSNR**     | Pixel-level differences | Noise or distortion in the image (objective).  |
| **SSIM**     | Structural similarity  | How similar the images look to a human viewer. |

### **Example Use Case:**  
- If you apply **Gaussian Blur** to an image:  
  - **PSNR** will drop significantly (high noise/blur).  
  - **SSIM** may decrease more gradually, reflecting how much structure is preserved.  
- If you apply **Median Filtering** (to reduce noise):  
  - **PSNR** improves if noise is reduced.  
  - **SSIM** might remain stable or slightly improve, as median filtering preserves edges better than Gaussian blur.

### **Summary:**  
- Use **PSNR** for quick, objective assessments of pixel-level quality.  
- Use **SSIM** to evaluate how well the image retains its visual structure and perceptual fidelity.  
- Together, these metrics provide a **comprehensive view** of image degradation or enhancement during processing.
---

## MIT License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

--- 
