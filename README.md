
# Image Noise Reducer with Streamlit

## Overview

This project is a simple image processing application built using Python and Streamlit. It allows users to upload an image, apply Gaussian Blur and Median Filter effects, and view the processed images. The application is powered by an `ImageProcessor` class that handles image loading, processing, and displaying.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.8 or higher
- Streamlit

You can install the required dependencies using pip:

```sh
pip install streamlit opencv-python matplotlib pillow
```

## Installation

1. **Clone the Repository**:
   - Clone this repository to your local machine using Git:

     ```sh
     git clone https://github.com/your-username/image-processing-streamlit.git
     ```

2. **Navigate to the Project Directory**:
   - Change directory to the project folder:

     ```sh
     cd image-processing-streamlit
     ```

3. **Install Dependencies**:
   - Install the required packages using pip:

     ```sh
     pip install streamlit opencv-python matplotlib pillow
     ```

## Running the Application

1. **Run Streamlit**:
   - Start the Streamlit application by running:

     ```sh
     streamlit run app.py
     ```

   - This command will open the application in your default web browser at `http://localhost:8501`.

2. **Upload an Image**:
   - Once the application is running, you will see a web interface where you can upload an image using the "Choose an Image..." file input.

3. **Process the Image**:
   - After uploading an image, the application will apply Gaussian Blur and Median Filter effects. The original and processed images will be displayed side by side.

4. **Save Processed Images**:
   - Click the "Save Images" button to save the processed images as `gaussian_blurred_image.jpg` and `median_filtered_image.jpg` in the current directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV: https://opencv.org/
- Streamlit: https://streamlit.io/


