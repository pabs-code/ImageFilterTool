import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

class ImageProcessor:
    """
    A class to process images using Gaussian Blur and Median Filter.
    
    Attributes:
        image_path (str): The path to the image file.
    """
    
    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with the path to an image.
        
        Args:
            image_path (str): The path to the image file.
        """
        self.image_path = image_path
        self.original_image = None
        self.gaussian_blur_image = None
        self.median_filter_image = None
    
    def load_image(self, image):
        """
        Load the image from the specified path.
        
        Raises:
            ValueError: If the image is not found.
        """
        self.original_image = np.array(image.convert('RGB'))
        
    def apply_gaussian_blur(self):
        """
        Apply Gaussian Blur to the image.
        
        Returns:
            numpy.ndarray: The blurred image.
        """
        if self.original_image is not None:
            self.gaussian_blur_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        return self.gaussian_blur_image
    
    def apply_median_filter(self):
        """
        Apply Median Filter to the image.
        
        Returns:
            numpy.ndarray: The filtered image.
        """
        if self.original_image is not None:
            self.median_filter_image = cv2.medianBlur(self.original_image, 5)
        return self.median_filter_image
    
    def display_images(self):
        """
        Display the original, Gaussian Blurred, and Median Filtered images.
        """
        return self.original_image, self.gaussian_blur_image, self.median_filter_image
    
    def save_images(self):
        """
        Save the Gaussian Blurred and Median Filtered images.
        """
        if self.gaussian_blur_image is not None and self.median_filter_image is not None:
            cv2.imwrite('gaussian_blurred_image.jpg', self.gaussian_blur_image)
            cv2.imwrite('median_filtered_image.jpg', self.median_filter_image)

def main():
    st.title("Image Processing with Streamlit")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        image_processor = ImageProcessor(None)  # Initialize with None since we'll load it later
        image_processor.load_image(image)
        
        gaussian_blur_image = image_processor.apply_gaussian_blur()
        median_filter_image = image_processor.apply_median_filter()
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_processor.original_image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        if gaussian_blur_image is not None:
            axs[1].imshow(gaussian_blur_image)
        else:
            axs[1].axis('off')
        
        if median_filter_image is not None:
            axs[2].imshow(median_filter_image)
        else:
            axs[2].axis('off')
        
        st.pyplot(fig)
        
        if st.button("Save Images"):
            image_processor.save_images()
            st.success("Images have been saved as 'gaussian_blurred_image.jpg' and 'median_filtered_image.jpg'.")

if __name__ == "__main__":
    main()
