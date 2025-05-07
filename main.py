import cv2
import matplotlib.pyplot as plt

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
    
    def load_image(self):
        """
        Load the image from the specified path.
        
        Raises:
            ValueError: If the image is not found.
        """
        self.original_image = cv2.imread(self.image_path)
        
        if self.original_image is None:
            raise ValueError("Image not found")
    
    def apply_gaussian_blur(self):
        """
        Apply Gaussian Blur to the image.
        
        Returns:
            numpy.ndarray: The blurred image.
        """
        self.gaussian_blur_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        return self.gaussian_blur_image
    
    def apply_median_filter(self):
        """
        Apply Median Filter to the image.
        
        Returns:
            numpy.ndarray: The filtered image.
        """
        self.median_filter_image = cv2.medianBlur(self.original_image, 5)
        return self.median_filter_image
    
    def display_images(self):
        """
        Display the original, Gaussian Blurred, and Median Filtered images.
        """
        plt.figure(figsize=(10, 7))
        
        # Display Original Image
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Display Gaussian Blurred Image
        plt.subplot(132)
        plt.title('Gaussian Blurred Image')
        plt.imshow(cv2.cvtColor(self.gaussian_blur_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Display Median Filtered Image
        plt.subplot(133)
        plt.title('Median Filtered Image')
        plt.imshow(cv2.cvtColor(self.median_filter_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()
    
    def save_images(self):
        """
        Save the Gaussian Blurred and Median Filtered images.
        """
        cv2.imwrite('gaussian_blurred_image.jpg', self.gaussian_blur_image)
        cv2.imwrite('median_filtered_image.jpg', self.median_filter_image)

def main():
    # Create an instance of ImageProcessor
    image_processor = ImageProcessor('./balloons_noisy.png')
    
    # Load the image
    image_processor.load_image()
    
    # Apply Gaussian Blur and Median Filter
    gaussian_blur_image = image_processor.apply_gaussian_blur()
    median_filter_image = image_processor.apply_median_filter()
    
    # Display the images
    image_processor.display_images()
    
    # Save the processed images
    image_processor.save_images()

if __name__ == "__main__":
    main()
