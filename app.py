import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


class ImageProcessor:
    """
    A class to handle the denoise processing of images using Gaussian Blur and Median Filter.

    Attributes:
        original_image (np.ndarray): The original image in RGB format.
        gaussian_blur_image (np.ndarray): Image after applying Gaussian blur.
        median_filter_image (np.ndarray): Image after applying median filter.
    """

    def __init__(self):
        self.original_image = None
        self.gaussian_blur_image = None
        self.median_filter_image = None

    def load_image(self, image: Image.Image) -> np.ndarray:
        """
        Load and convert uploaded image to RGB format.

        Args:
            image (PIL.Image): The uploaded image.

        Returns:
            numpy.ndarray: Processed image as a NumPy array.
        """
        self.original_image = np.array(image).astype("uint8")
        return self.original_image

    def apply_gaussian_blur(
        self, ksize: tuple[int, int] = (5, 5), sigma: float = 0.0
    ) -> np.ndarray:
        """
        Apply Gaussian Blur with parameterized kernel size and sigma.

        Args:
            ksize (tuple): Kernel size as a tuple (height, width).
            sigma (float): Standard deviation for the Gaussian kernel.

        Returns:
            numpy.ndarray: Processed image after applying Gaussian blur.
        """
        if self.original_image is not None:
            self.gaussian_blur_image = cv2.GaussianBlur(
                self.original_image, ksize, sigma
            )
        return self.gaussian_blur_image

    def apply_median_filter(self, ksize: int = 5) -> np.ndarray:
        """
        Apply Median Filter with parameterized kernel size.

        Args:
            ksize (int): Kernel size for the median filter.

        Returns:
            numpy.ndarray: Processed image after applying median filtering.
        """
        if self.original_image is not None:
            self.median_filter_image = cv2.medianBlur(
                self.original_image, ksize
            )
        return self.median_filter_image

    def save_images(self):
        """
        Save processed images in correct color space (BGR for OpenCV).
        """
        if self.gaussian_blur_image is not None:
            cv2.imwrite("gaussian_blurred_image.jpg", cv2.cvtColor(
                self.gaussian_blur_image, cv2.COLOR_RGB2BGR))
        if self.median_filter_image is not None:
            cv2.imwrite("median_filtered_image.jpg", cv2.cvtColor(
                self.median_filter_image, cv2.COLOR_RGB2BGR))

    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).

        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.

        Returns:
            float: PSNR value in dB.
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.

        Returns:
            float: SSIM value.
        """
        if img1 is None or img2 is None:
            return 0.0

        # Ensure both images are of the same shape
        if img1.shape != img2.shape:
            return 0.0

        min_dim = min(img1.shape[:2])  # height, width
        if min_dim < 7:
            return 0.0

        win_size = min(11, min_dim)
        if win_size % 2 == 0:
            win_size -= 1

        try:
            # Ensure that channel_axis is set properly
            return ssim(img1, img2, win_size=win_size, multichannel=True, channel_axis=-1)
        except Exception as e:
            print(f"SSIM calculation failed: {e}")
            return 0.0


def main():
    st.set_page_config(
        page_title="Image Denoise Filter Tool App", layout="wide")
    st.title("🖼️ Image Denoise Filter Tool with Streamlit")
    st.markdown(
        "### Instructions:\n\n"
        "- Upload an image file: (jpg, jpeg, png)\n"
        "- Adjust the parameters for Gaussian Blur, Median Filter Kernel Size and Gaussian Sigma.\n"
        "- View processed images, metrics (PSNR & SSIM), and histograms of RGB color channels for each image.\n"
        "- You can click ```Save Processed Images``` to save both results as JPEG files.\n"
    )

    uploaded_file = st.file_uploader(
        "Upload an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            processor = ImageProcessor()
            original_image = processor.load_image(image)

            st.markdown("### 🔧 Adjust Image Filter Parameters")
            col1, col2 = st.columns(2)
            with col1:
                gauss_ksize = st.slider(
                    "Gaussian Blur Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2
                )
                gauss_sigma = st.slider(
                    "Gaussian Sigma",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.5
                )
            with col2:
                median_ksize = st.slider(
                    "Median Filter Kernel Size",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2
                )

            st.markdown("### 📈 Processed Results")
            gaussian_image = processor.apply_gaussian_blur(
                (gauss_ksize, gauss_ksize), gauss_sigma)
            median_image = processor.apply_median_filter(median_ksize)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original_image, caption="Original",
                         use_container_width=True)
            with col2:
                st.image(gaussian_image, caption="Gaussian Blur",
                         use_container_width=True)
            with col3:
                st.image(median_image, caption="Median Filter",
                         use_container_width=True)

            st.markdown("### 📊 Metrics")
            psnr_gauss = processor.compute_psnr(original_image, gaussian_image)
            psnr_median = processor.compute_psnr(original_image, median_image)

            ssim_gauss = processor.compute_ssim(original_image, gaussian_image)
            ssim_median = processor.compute_ssim(original_image, median_image)

            metrics_table = f"""
                | Metric         | Gaussian Blur     | Median Filter    |
                |----------------|------------------|------------------|
                | PSNR           | {psnr_gauss:.2f} dB  | {psnr_median:.2f} dB |
                | SSIM           | {ssim_gauss:.4f}   | {ssim_median:.4f}  |
            """

            st.markdown(metrics_table, unsafe_allow_html=True)

            if st.button("Save Processed Images"):
                processor.save_images()
                st.success(
                    "Images saved to disk as 'gaussian_blurred_image.jpg' and 'median_filtered_image.jpg'.")

            st.markdown("### 📊 Histograms of RGB Channels")
            if st.checkbox("Show Color Histograms"):
                fig, axs = plt.subplots(3, 3, figsize=(12, 9))
                channels = ["Red", "Green", "Blue"]

                for i, ax in enumerate(axs[0]):
                    channel_data = original_image[:, :, [2, 1, 0]][i]
                    color = ['red', 'green', 'blue'][i]
                    ax.hist(channel_data.flatten(), bins=256,
                            facecolor=color, alpha=0.5)
                    ax.set_title(f"Original - {channels[i]}")

                for i, ax in enumerate(axs[1]):
                    channel_data = gaussian_image[:, :, [2, 1, 0]][i]
                    color = ['red', 'green', 'blue'][i]
                    ax.hist(channel_data.flatten(), bins=256,
                            facecolor=color, alpha=0.5)
                    ax.set_title(f"Gaussian - {channels[i]}")

                for i, ax in enumerate(axs[2]):
                    channel_data = median_image[:, :, [2, 1, 0]][i]
                    color = ['red', 'green', 'blue'][i]
                    ax.hist(channel_data.flatten(), bins=256,
                            facecolor=color, alpha=0.5)
                    ax.set_title(f"Median - {channels[i]}")

                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
